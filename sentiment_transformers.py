""" self-contained sentiment analysis model finetuning on hugginface.transformers (imdb/sst) """
import argparse
import os
import random
import json
import logging
from logging.config import dictConfig
from time import time

import transformers
import torch
from torch import optim
from torch.autograd import detect_anomaly
from torch.utils.tensorboard import SummaryWriter

from get_dataset import get_dataset_sentiment
from checkpoint_versioning import Argument


dictConfig({
    "version": 1,
    "formatters": {'f': {'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}},
    "handlers": {'h': {'class': 'logging.StreamHandler', 'formatter': 'f', 'level': logging.DEBUG}},
    "root": {'handlers': ['h'], 'level': logging.DEBUG}})
LOGGER = logging.getLogger()
NUM_WORKER = int(os.getenv("NUM_WORKER", '4'))
PROGRESS_INTERVAL = int(os.getenv("PROGRESS_INTERVAL", '100'))
CACHE_DIR = os.getenv("CACHE_DIR", './cache')
CKPT_DIR = os.getenv("CKPT_DIR", './ckpt')


class Transforms:
    """ Text encoder with transformers tokenizer """

    def __init__(self, transformer_tokenizer: str, max_seq_length: int = None):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(transformer_tokenizer, cache_dir=CACHE_DIR)
        if max_seq_length and max_seq_length > self.tokenizer.max_len:
            raise ValueError('`max_seq_length should be less than %i' % self.tokenizer.max_len)
        self.max_seq_length = max_seq_length if max_seq_length else self.tokenizer.max_len

    def __call__(self, text: str):
        return self.tokenizer.encode_plus(text, max_length=self.max_seq_length, pad_to_max_length=True)


class Dataset(torch.utils.data.Dataset):
    """ torch.utils.data.Dataset with transformer tokenizer """

    def __init__(self, data: list, transform_function, label: list = None):
        self.data = data  # list of half-space split tokens
        self.label = label if label is None else [int(l) for l in label]  # list of label sequence
        self.transform_function = transform_function

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encode = self.transform_function(self.data[idx])
        if self.label is not None:
            encode['labels'] = self.label[idx]
        float_list = ['attention_mask']
        encode = {k: torch.tensor(v, dtype=torch.float32) if k in float_list else torch.tensor(v, dtype=torch.long)
                  for k, v in encode.items()}
        return encode


class TransformerSequenceClassification:
    """ finetune transformers on text classification """

    def __init__(self, dataset: str, batch_size_validation: int = None, checkpoint: str = None, **kwargs):
        LOGGER.info('*** initialize network ***')

        # checkpoint version
        self.args = Argument(prefix=dataset, checkpoint=checkpoint, dataset=dataset, **kwargs)
        self.batch_size_validation = batch_size_validation if batch_size_validation else self.args.batch_size

        # fix random seed
        random.seed(self.args.random_seed)
        transformers.set_seed(self.args.random_seed)
        torch.manual_seed(self.args.random_seed)
        torch.cuda.manual_seed_all(self.args.random_seed)

        # model setup
        ckpt_statistics = self.load_ckpt()
        if ckpt_statistics:
            stats, self.label_to_id = ckpt_statistics
            self.dataset_split = None
        else:
            stats = None
            self.dataset_split, self.label_to_id = get_dataset_sentiment(self.args.dataset)
            with open(os.path.join(self.args.checkpoint_dir, 'label_to_id.json'), 'w') as f:
                json.dump(self.label_to_id, f)
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            self.args.transformer,
            config=transformers.AutoConfig.from_pretrained(
                self.args.transformer,
                num_labels=len(self.id_to_label),
                id2label=self.id_to_label,
                label2id=self.label_to_id,
                cache_dir=CACHE_DIR)
        )
        self.transforms = Transforms(self.args.transformer, self.args.max_seq_length)

        # optimizer
        if self.args.optimizer == 'adamw':
            self.optimizer = transformers.AdamW(
                self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            raise ValueError('bad optimizer: %s' % self.args.optimizer)

        # scheduler
        if self.args.scheduler == 'constant':
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1, last_epoch=-1)
        elif self.args.scheduler == 'linear':
            self.scheduler = transformers.get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.warmup_step,
                num_training_steps=self.args.total_step)
        else:
            raise ValueError('unknown scheduler: %s' % self.args.scheduler)

        # load checkpoint
        self.__step = 0 if stats is None else stats['step']  # num of training step
        self.__epoch = 0 if stats is None else stats['epoch']  # num of epoch
        self.__best_val_score = None if stats is None else stats['best_val_score']

        # apply checkpoint statistics to optimizer/scheduler
        if stats is not None:
            self.model.load_state_dict(stats['model_state_dict'])
            if self.optimizer is not None and self.scheduler is not None:
                self.optimizer.load_state_dict(stats['optimizer_state_dict'])
                self.scheduler.load_state_dict(stats['scheduler_state_dict'])

        # GPU allocation
        self.n_gpu = torch.cuda.device_count()
        self.device = 'cuda' if self.n_gpu > 0 else 'cpu'
        self.model.to(self.device)

        # GPU mixture precision
        self.scale_loss = None
        if self.args.fp16:
            try:
                from apex import amp  # noqa: F401
                self.model, self.optimizer = amp.initialize(
                    self.model, self.optimizer, opt_level='O1')
                self.scale_loss = amp.scale_loss
                LOGGER.info('using `apex.amp`')
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        # multi-gpus
        if self.n_gpu > 1:
            # multi-gpu training (should be after apex fp16 initialization)
            self.model = torch.nn.DataParallel(self.model.cuda())
            LOGGER.info('using `torch.nn.DataParallel`')
        LOGGER.info('running on %i GPUs' % self.n_gpu)

    def load_ckpt(self):
        checkpoint_file = os.path.join(self.args.checkpoint_dir, 'model.pt')
        label_id_file = os.path.join(self.args.checkpoint_dir, 'label_to_id.json')
        if os.path.exists(checkpoint_file):
            assert os.path.exists(label_id_file)
            LOGGER.info('load ckpt from %s' % checkpoint_file)
            ckpt = torch.load(checkpoint_file, map_location='cpu')  # allocate stats on cpu
            ckpt_dict = {
                "step": ckpt['step'],
                "epoch": ckpt['epoch'],
                "model_state_dict": ckpt['model_state_dict'],
                "best_val_score": ckpt['best_val_score'],
                "optimizer_state_dict": ckpt['optimizer_state_dict'],
                "scheduler_state_dict": ckpt['scheduler_state_dict']
            }
            label_to_id = json.load(open(label_id_file, 'r'))
            return ckpt_dict, label_to_id
        else:
            return None

    def predict(self, x: list):
        """ model inference """
        self.model.eval()
        data_loader = torch.utils.data.DataLoader(
            Dataset(x, transform_function=self.transforms), num_workers=NUM_WORKER, batch_size=len(x))
        prediction, prob = [], []
        for encode in data_loader:
            logit = self.model(**{k: v.to(self.device) for k, v in encode.items()})[0]
            _, _pred = torch.max(logit, dim=1)
            _pred_list = _pred.cpu().tolist()
            _prob_list = torch.nn.functional.softmax(logit, dim=1).cpu().tolist()
            prediction += [self.id_to_label[_p] for _p in _pred_list]
            prob += [dict([(self.id_to_label[i], float(pr)) for i, pr in enumerate(_p)]) for _p in _prob_list]
        return prediction, prob

    def test(self):
        LOGGER.addHandler(logging.FileHandler(os.path.join(self.args.checkpoint_dir, 'logger_test.log')))
        if self.dataset_split is None:
            self.dataset_split = get_dataset_sentiment(
                self.args.dataset, label_to_id=self.label_to_id, allow_update=False)
        data_loader_test = {k: torch.utils.data.DataLoader(
            Dataset(**v, transform_function=self.transforms),
            num_workers=NUM_WORKER,
            batch_size=self.args.batch_size)
            for k, v in self.dataset_split.items() if k not in ['train', 'valid']}
        LOGGER.info('data_loader_test: %s' % str(list(data_loader_test.keys())))
        assert len(data_loader_test.keys()) != 0, 'no test set found'
        start_time = time()
        writer = SummaryWriter(log_dir=self.args.checkpoint_dir)
        for k, v in data_loader_test.items():
            self.__epoch_valid(v, writer=writer, prefix=k)
            self.release_cache()
        writer.close()
        LOGGER.info('[test completed, %0.2f sec in total]' % (time() - start_time))

    def train(self):
        LOGGER.addHandler(logging.FileHandler(os.path.join(self.args.checkpoint_dir, 'logger_train.log')))
        if self.dataset_split is None:
            self.dataset_split = get_dataset_sentiment(
                self.args.dataset, label_to_id=self.label_to_id, allow_update=False)
        writer = SummaryWriter(log_dir=self.args.checkpoint_dir)
        start_time = time()
        data_loader = {k: torch.utils.data.DataLoader(
            Dataset(**self.dataset_split.pop(k), transform_function=self.transforms),
            num_workers=NUM_WORKER,
            batch_size=self.args.batch_size if k == 'train' else self.batch_size_validation,
            shuffle=k == 'train',
            drop_last=k == 'train')
            for k in ['train', 'valid']}
        LOGGER.info('data_loader: %s' % str(list(data_loader.keys())))
        LOGGER.info('*** start training from step %i, epoch %i ***' % (self.__step, self.__epoch))
        try:
            with detect_anomaly():
                while True:
                    if_training_finish = self.__epoch_train(data_loader['train'], writer=writer)
                    self.release_cache()
                    if_early_stop = self.__epoch_valid(data_loader['valid'], writer=writer, prefix='valid')
                    self.release_cache()
                    if if_training_finish or if_early_stop:
                        break
                    self.__epoch += 1
        except RuntimeError:
            LOGGER.exception('*** RuntimeError (NaN found, see above log in detail) ***')

        except KeyboardInterrupt:
            LOGGER.info('*** KeyboardInterrupt ***')

        if self.__best_val_score is None:
            self.args.remove_ckpt()
            exit('nothing to be saved')

        LOGGER.info('[training completed, %0.2f sec in total]' % (time() - start_time))
        if self.n_gpu > 1:
            model_wts = self.model.module.state_dict()
        else:
            model_wts = self.model.state_dict()
        torch.save({
            'step': self.__step,
            'epoch': self.__epoch,
            'model_state_dict': model_wts,
            'best_val_score': self.__best_val_score,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, os.path.join(self.args.checkpoint_dir, 'model.pt'))
        writer.close()
        LOGGER.info('ckpt saved at %s' % self.args.checkpoint_dir)

    def release_cache(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def __epoch_train(self, data_loader, writer):
        """ train on single epoch return flag which is True if training has been completed """
        self.model.train()
        for i, encode in enumerate(data_loader, 1):
            # update model
            encode = {k: v.to(self.device) for k, v in encode.items()}
            self.optimizer.zero_grad()
            loss, logit = self.model(**encode)[0:2]
            if self.n_gpu > 1:
                loss = loss.mean()
            if self.args.fp16:
                with self.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            # optimizer and scheduler step
            self.optimizer.step()
            self.scheduler.step()
            # log instantaneous accuracy, loss, and learning rate
            _, pred = torch.max(logit, 1)
            inst_accuracy = ((pred == encode['labels']).cpu().float().mean()).item()
            inst_loss = loss.cpu().detach().item()
            inst_lr = self.optimizer.param_groups[0]['lr']
            writer.add_scalar('train/loss', inst_loss, self.__step)
            writer.add_scalar('train/learning_rate', inst_lr, self.__step)
            writer.add_scalar('train/accuracy', inst_accuracy, self.__step)
            if self.__step % PROGRESS_INTERVAL == 0:
                LOGGER.info('[epoch %i] * (training step %i) loss: %.3f, lr: %0.8f'
                            % (self.__epoch, self.__step, inst_loss, inst_lr))
            self.__step += 1
            # break
            if self.__step >= self.args.total_step:
                LOGGER.info('reached maximum step')
                return True
        return False

    def __epoch_valid(self, data_loader, writer, prefix: str='valid'):
        """ validation/test """
        self.model.eval()
        list_accuracy, list_loss = [], []
        for encode in data_loader:
            encode = {k: v.to(self.device) for k, v in encode.items()}
            loss, logit = self.model(**encode)[0:2]
            if self.n_gpu > 1:
                loss = torch.sum(loss)
            _, pred = torch.max(logit, 1)
            list_accuracy.append(((pred == encode['labels']).cpu().float().mean()).item())
            list_loss.append(loss.cpu().item())
        accuracy, loss = float(sum(list_accuracy)/len(list_accuracy)), float(sum(list_loss)/len(list_loss))
        LOGGER.info('[epoch %i] (%s) accuracy: %.3f, loss: %.3f' % (self.__epoch, prefix, accuracy, loss))
        writer.add_scalar('%s/accuracy' % prefix, accuracy, self.__epoch)
        writer.add_scalar('%s/loss' % prefix, loss, self.__epoch)
        if prefix == 'valid':
            if self.__best_val_score is None or accuracy > self.__best_val_score:
                self.__best_val_score = accuracy
            if self.args.early_stop and self.__best_val_score - accuracy > self.args.early_stop:
                return True
        return False


def get_options():
    parser = argparse.ArgumentParser(
        description='finetune transformers to sentiment analysis',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c', '--checkpoint', help='checkpoint to load', default=None, type=str)
    parser.add_argument('-d', '--data', help='data sst/imdb', default='sst', type=str)
    parser.add_argument('-t', '--transformer', help='pretrained language model', default='xlm-roberta-base', type=str)
    parser.add_argument('-m', '--max-seq-length',
                        help='max sequence length (use same length as used in pre-training if not provided)',
                        default=128,
                        type=int)
    parser.add_argument('-b', '--batch-size', help='batch size', default=32, type=int)
    parser.add_argument('--random-seed', help='random seed', default=1234, type=int)
    parser.add_argument('--lr', help='learning rate', default=1e-5, type=float)
    parser.add_argument('--optimizer', help='optimizer', default='adamw', type=str)
    parser.add_argument('--scheduler', help='scheduler', default='linear', type=str)
    parser.add_argument('--total-step', help='total training step', default=13000, type=int)
    parser.add_argument('--batch-size-validation',
                        help='batch size for validation (smaller size to save memory)',
                        default=1,
                        type=int)
    parser.add_argument('--warmup-step', help='warmup step (6 percent of total is recommended)', default=700, type=int)
    parser.add_argument('--weight-decay', help='weight decay', default=1e-7, type=float)
    parser.add_argument('--early-stop', help='value of accuracy drop for early stop', default=0.1, type=float)
    parser.add_argument('--inference-mode', help='inference mode', action='store_true')
    parser.add_argument('--test', help='run over testdataset', action='store_true')
    parser.add_argument('--fp16', help='fp16', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    opt = get_options()
    classifier = TransformerSequenceClassification(
        batch_size_validation=opt.batch_size_validation,
        checkpoint=opt.checkpoint,
        dataset=opt.data,
        transformer=opt.transformer,
        random_seed=opt.random_seed,
        lr=opt.lr,
        optimizer=opt.optimizer,
        scheduler=opt.scheduler,
        total_step=opt.total_step,
        warmup_step=opt.warmup_step,
        weight_decay=opt.weight_decay,
        batch_size=opt.batch_size,
        max_seq_length=opt.max_seq_length,
        early_stop=opt.early_stop,
        fp16=opt.fp16
    )
    if opt.inference_mode:
        while True:
            _inp = input('input sentence >>>')
            if _inp == 'q':
                break
            elif _inp == '':
                continue
            else:
                predictions, probs = classifier.predict([_inp])
                print(predictions)
                print(probs)
    else:
        if opt.test:
            classifier.test()
        else:
            classifier.train()

