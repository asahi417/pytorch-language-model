""" self-contained NER finetuning on hugginface.transformers (conll_2003/wnut_17) """

import traceback
import argparse
import os
import random
import hashlib
import json
import logging
import shutil
import transformers
import torch
from time import time
from torch import optim
from torch import nn
from torch.autograd import detect_anomaly
from torch.utils.tensorboard import SummaryWriter
from glob import glob
from logging.config import dictConfig
from itertools import chain
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report, accuracy_score


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
PAD_TOKEN_LABEL_ID = nn.CrossEntropyLoss().ignore_index


def get_dataset(data_name: str = 'wnut_17', label_to_id: dict = None):
    """ download dataset file and return dictionary including training/validation split """
    label_to_id = dict() if label_to_id is None else label_to_id
    data_path = os.path.join(CACHE_DIR, data_name)

    def decode_file(file_name, _label_to_id: dict):
        inputs, labels = [], []
        with open(os.path.join(data_path, file_name), 'r') as f:
            sentence, entity = [], []
            for n, line in enumerate(f):
                line = line.strip()
                if len(line) == 0 or line.startswith("-DOCSTART-"):
                    if len(sentence) != 0:
                        assert len(sentence) == len(entity)
                        inputs.append(sentence)
                        labels.append(entity)
                        sentence, entity = [], []
                else:
                    ls = line.split()
                    sentence.append(ls[0])
                    # Examples could have no label for mode = "test"
                    tag = 'O' if len(ls) < 2 else ls[-1]
                    if tag not in _label_to_id.keys():
                        _label_to_id[tag] = len(_label_to_id)
                    entity.append(_label_to_id[tag])
        return _label_to_id, {"data": inputs, "label": labels}

    if data_name == 'conll_2003':
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            os.system('git clone https://github.com/mohammadKhalifa/xlm-roberta-ner')
            os.system('mv ./xlm-roberta-ner/data/coNLL-2003/* %s/' % data_path)
            os.system('rm -rf ./xlm-roberta-ner')
        files = ['train.txt', 'valid.txt', 'test.txt']
    elif data_name == 'wnut_17':
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            os.system("curl -L 'https://github.com/leondz/emerging_entities_17/raw/master/wnut17train.conll'  | tr '\t' ' ' > %s/train.txt.tmp" % data_path)
            os.system("curl -L 'https://github.com/leondz/emerging_entities_17/raw/master/emerging.dev.conll' | tr '\t' ' ' > %s/dev.txt.tmp" % data_path)
            os.system("curl -L 'https://raw.githubusercontent.com/leondz/emerging_entities_17/master/emerging.test.annotated' | tr '\t' ' ' > %s/test.txt.tmp" % data_path)
        files = ['train.txt.tmp', 'dev.txt.tmp', 'test.txt.tmp']
    else:
        raise ValueError('unknown dataset: %s' % data_name)

    data_split = dict()
    for name, filepath in zip(['train', 'valid', 'test'], files):
        label_to_id, data_dict = decode_file(filepath, _label_to_id=label_to_id)
        data_split[name] = data_dict
        LOGGER.info('dataset %s/%s: %i entries' % (data_name, filepath, len(data_dict['data'])))
    return data_split, label_to_id


class Argument:
    """ Model training arguments manager """

    def __init__(self, prefix: str = None, checkpoint: str = None, **kwargs):
        """  Model training arguments manager

         Parameter
        -------------------
        prefix: prefix to filename
        checkpoint: existing checkpoint name if you want to load
        kwargs: model arguments
        """
        self.checkpoint_dir, self.parameter = self.__version(kwargs, checkpoint, prefix)
        LOGGER.info('checkpoint: %s' % self.checkpoint_dir)
        for k, v in self.parameter.items():
            LOGGER.info(' - [arg] %s: %s' % (k, str(v)))
        self.__dict__.update(self.parameter)

    def remove_ckpt(self):
        shutil.rmtree(self.checkpoint_dir)

    @staticmethod
    def md5(file_name):
        """ get MD5 checksum """
        hash_md5 = hashlib.md5()
        with open(file_name, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def __version(self, parameter: dict = None, checkpoint: str = None, prefix: str = None):
        """ Checkpoint version

         Parameter
        ---------------------
        parameter: parameter configuration to find same setting checkpoint
        checkpoint: existing checkpoint to be loaded

         Return
        --------------------
        path_to_checkpoint: path to new checkpoint dir
        parameter: parameter
        """

        if checkpoint is None and parameter is None:
            raise ValueError('either of `checkpoint` or `parameter` is needed.')

        if checkpoint is None:
            LOGGER.info('issue new checkpoint id')
            # check if there are any checkpoints with same hyperparameters
            version_name = []
            for parameter_path in glob(os.path.join(CKPT_DIR, '*/parameter.json')):
                _dir = parameter_path.replace('/parameter.json', '')
                _dict = json.load(open(parameter_path))
                version_name.append(_dir.split('/')[-1])
                if parameter == _dict:
                    inp = input('found a checkpoint with same configuration\n'
                                'enter to delete the existing checkpoint %s\n'
                                'or exit by type anything but not empty' % _dir)
                    if inp == '':
                        shutil.rmtree(_dir)
                    else:
                        exit()

            with open(os.path.join(CKPT_DIR, 'tmp.json'), 'w') as _f:
                json.dump(parameter, _f)
            new_checkpoint = self.md5(os.path.join(CKPT_DIR, 'tmp.json'))
            new_checkpoint = '_'.join([prefix, new_checkpoint]) if prefix else new_checkpoint
            new_checkpoint_dir = os.path.join(CKPT_DIR, new_checkpoint)
            os.makedirs(new_checkpoint_dir, exist_ok=True)
            shutil.move(os.path.join(CKPT_DIR, 'tmp.json'), os.path.join(new_checkpoint_dir, 'parameter.json'))
            return new_checkpoint_dir, parameter

        else:
            LOGGER.info('load existing checkpoint')
            checkpoints = glob(os.path.join(CKPT_DIR, checkpoint, 'parameter.json'))
            if len(checkpoints) >= 2:
                raise ValueError('Checkpoints are duplicated: %s' % str(checkpoints))
            elif len(checkpoints) == 0:
                raise ValueError('No checkpoint: %s' % os.path.join(CKPT_DIR, checkpoint))
            else:
                parameter = json.load(open(checkpoints[0]))
                target_checkpoints_path = checkpoints[0].replace('/parameter.json', '')
                return target_checkpoints_path, parameter


class Transforms:
    """ Text encoder with transformers tokenizer """

    def __init__(self, transformer_tokenizer: str, max_seq_length: int = None, pad_to_max_length: bool = True):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(transformer_tokenizer, cache_dir=CACHE_DIR)
        if max_seq_length and max_seq_length > self.tokenizer.max_len:
            raise ValueError('`max_seq_length should be less than %i' % self.tokenizer.max_len)
        self.max_seq_length = max_seq_length if max_seq_length else self.tokenizer.max_len
        self.pad_to_max_length = pad_to_max_length

    def __call__(self, text: str):
        return self.tokenizer.encode_plus(
            text, max_length=self.max_seq_length, pad_to_max_length=self.pad_to_max_length)

    @property
    def all_special_ids(self):
        return self.tokenizer.all_special_ids

    def tokenize(self, *args, **kwargs):
        return self.tokenizer.tokenize(*args, **kwargs)


class Dataset(torch.utils.data.Dataset):
    """ torch.utils.data.Dataset with transformer tokenizer """

    def __init__(self, data: list, transform_function, label: list = None):
        self.data = data  # list of half-space split tokens
        self.transform_function = transform_function
        self.label = self.fix_label(label, data, transform_function.tokenize, transform_function.all_special_ids)

    @staticmethod
    def fix_label(self, label, data, tokenizer, all_special_ids):
        assert len(label) == len(data)
        fixed_labels = []
        for x, y in zip(label, data):
            assert len(y) == len(x)
            encode = self.transform_function(' '.join(x))
            fixed_label = list(chain(*[
                [label] + [PAD_TOKEN_LABEL_ID] * (len(tokenizer(word)) - 1)
                for label, word in zip(y, x)]))
            if encode['input_ids'][0] in all_special_ids:
                fixed_label = [PAD_TOKEN_LABEL_ID] + fixed_label
            fixed_label += [PAD_TOKEN_LABEL_ID] * (len(encode['input_ids']) - len(fixed_label))
            fixed_labels.append(fixed_label)
        return fixed_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encode = self.transform_function(' '.join(self.data[idx]))
        if self.label is not None:
            encode['labels'] = self.label[idx]
        float_list = ['attention_mask']
        encode = {k: torch.tensor(v, dtype=torch.float32) if k in float_list else torch.tensor(v, dtype=torch.long)
                  for k, v in encode.items()}
        return encode


class TransformerTokenClassification:
    """ finetune transformers on token classification """

    def __init__(self, dataset: str, batch_size_validation: int = None,
                 checkpoint: str = None, inference_mode: bool = False, **kwargs):
        self.inference_mode = inference_mode
        LOGGER.info('*** initialize network (INFERENCE MODE: %s) ***' % str(self.inference_mode))

        # checkpoint version
        self.args = Argument(prefix=dataset, checkpoint=checkpoint, dataset=dataset, **kwargs)
        self.batch_size_validation = batch_size_validation if batch_size_validation else self.args.batch_size

        # fix random seed
        random.seed(self.args.random_seed)
        transformers.set_seed(self.args.random_seed)
        torch.manual_seed(self.args.random_seed)
        torch.cuda.manual_seed_all(self.args.random_seed)

        # model/dataset setup
        stats, label_to_id = self.load_ckpt()
        if self.inference_mode:
            if stats is None or label_to_id is None:
                raise ValueError('As no checkpoints found, unable to perform inference.')
            self.dataset_split, self.label_to_id = None, label_to_id
            self.writer = None
        else:
            self.dataset_split, self.label_to_id = get_dataset(self.args.dataset, label_to_id=label_to_id)
            self.writer = SummaryWriter(log_dir=self.args.checkpoint_dir)
        self.id_to_label = {v: str(k) for k, v in self.label_to_id.items()}
        self.model = transformers.AutoModelForTokenClassification.from_pretrained(
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
        if self.inference_mode:
            self.optimizer = self.scheduler = None
        else:
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
                    self.model, self.optimizer, opt_level='O1', max_loss_scale=2**13, min_loss_scale=1e-5)
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
            return None, None

    # def predict(self, x: list):
    #     """ model inference """
    #     self.model.eval()
    #     encode = self.transforms(x)
    #     encode = {k: torch.tensor(v, dtype=torch.long).to(self.device) for k, v in encode.items()}
    #     logit = self.model(**encode)[0]
    #     pred = torch.max(logit, 2)[1].cpu().detach().int().tolist()
    #     prediction = [[self.id_to_label[_p] for _p in batch] for batch in pred]
    #     return prediction

    def train(self):
        LOGGER.addHandler(logging.FileHandler(os.path.join(self.args.checkpoint_dir, 'logger.log')))
        if self.inference_mode:
            raise ValueError('model is on an inference mode')

        LOGGER.info('*** start training from step %i, epoch %i ***' % (self.__step, self.__epoch))
        start_time = time()
        data_loader = {k: torch.utils.data.DataLoader(
            Dataset(**self.dataset_split.pop(k), transform_function=self.transforms),
            num_workers=NUM_WORKER,
            batch_size=self.args.batch_size if k == 'train' else self.batch_size_validation,
            shuffle=k == 'train',
            drop_last=k == 'train')
            for k in ['train', 'valid']}
        # data_loader_test = {k: torch.utils.data.DataLoader(
        #     Dataset(**v, **shared),
        #     num_workers=NUM_WORKER,
        #     batch_size=self.batch_size_validation)
        #     for k, v in self.dataset_split.items()}
        LOGGER.info('data_loader     : %s' % str(list(data_loader.keys())))
        # LOGGER.info('data_loader_test: %s' % str(list(data_loader_test.keys())))
        try:
            with detect_anomaly():
                while True:
                    if_training_finish = self.__epoch_train(data_loader['train'])
                    if_early_stop = self.__epoch_valid(data_loader['valid'], prefix='valid')
                    if if_training_finish or if_early_stop:
                        break
                    self.__epoch += 1
                # for k, v in data_loader.items():
                #     self.__epoch_valid(v, prefix=k)

        except RuntimeError:
            LOGGER.info(traceback.format_exc())
            LOGGER.info('*** RuntimeError (NaN found, see above log in detail) ***')

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
        with open(os.path.join(self.args.checkpoint_dir, 'label_to_id.json'), 'w') as f:
            json.dump(self.label_to_id, f)
        self.writer.close()
        LOGGER.info('ckpt saved at %s' % self.args.checkpoint_dir)

    def __epoch_train(self, data_loader):
        """ train on single epoch, returning flag which is True if training has been completed """
        self.model.train()
        for i, encode in enumerate(data_loader, 1):
            # update model
            encode = {k: v.to(self.device) for k, v in encode.items()}
            self.optimizer.zero_grad()
            loss = self.model(**encode)[0]
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
            inst_loss = loss.cpu().detach().item()
            inst_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('train/loss', inst_loss, self.__step)
            self.writer.add_scalar('train/learning_rate', inst_lr, self.__step)
            if self.__step % PROGRESS_INTERVAL == 0:
                LOGGER.info('[epoch %i] * (training step %i) loss: %.3f, lr: %0.8f'
                            % (self.__epoch, self.__step, inst_loss, inst_lr))
            self.__step += 1
            # break
            if self.__step >= self.args.total_step:
                LOGGER.info('reached maximum step')
                return True

        return False

    def __epoch_valid(self, data_loader, prefix: str='valid'):
        """ validation/test, returning flag which is True if early stop condition was applied """
        self.model.eval()
        list_loss, seq_pred, seq_true = [], [], []
        for encode in data_loader:
            encode = {k: v.to(self.device) for k, v in encode.items()}
            model_outputs = self.model(**encode)
            loss, logit = model_outputs[0:2]
            if self.n_gpu > 1:
                loss = torch.sum(loss)
            list_loss.append(loss.cpu().detach().item())
            _true = encode['labels'].cpu().detach().int().tolist()
            _pred = torch.max(logit, 2)[1].cpu().detach().int().tolist()
            for b in range(len(_true)):
                _pred_list, _true_list = [], []
                for s in range(len(_true[b])):
                    if _true[b][s] != PAD_TOKEN_LABEL_ID:
                        _true_list.append(self.id_to_label[_pred[b][s]])
                        _pred_list.append(self.id_to_label[_true[b][s]])
                assert len(_pred_list) == len(_true_list)
                if len(_true_list) > 0:
                    seq_true.append(_true_list)
                    seq_pred.append(_pred_list)

        LOGGER.info('[epoch %i] (%s) \n %s' % (self.__epoch, prefix, classification_report(seq_true, seq_pred)))
        self.writer.add_scalar('%s/f1' % prefix, f1_score(seq_true, seq_pred), self.__epoch)
        self.writer.add_scalar('%s/recall' % prefix, recall_score(seq_true, seq_pred), self.__epoch)
        self.writer.add_scalar('%s/precision' % prefix, precision_score(seq_true, seq_pred), self.__epoch)
        self.writer.add_scalar('%s/accuracy' % prefix, accuracy_score(seq_true, seq_pred), self.__epoch)
        self.writer.add_scalar('%s/loss' % prefix, float(sum(list_loss) / len(list_loss)), self.__epoch)
        if prefix == 'valid':
            score = f1_score(seq_true, seq_pred)
            if self.__best_val_score is None or score > self.__best_val_score:
                self.__best_val_score = score
            if self.args.early_stop and self.__best_val_score - score > self.args.early_stop:
                return True
        return False


def get_options():
    parser = argparse.ArgumentParser(
        description='finetune transformers to sentiment analysis',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c', '--checkpoint', help='checkpoint to load', default=None, type=str)
    parser.add_argument('-d', '--data', help='data conll_2003/wnut_17', default='wnut_17', type=str)
    parser.add_argument('-t', '--transformer', help='pretrained language model', default='xlm-roberta-base', type=str)
    parser.add_argument('-m', '--max-seq-length',
                        help='max sequence length (use same length as used in pre-training if not provided)',
                        default=128,
                        type=int)
    parser.add_argument('-b', '--batch-size', help='batch size', default=16, type=int)
    parser.add_argument('--random-seed', help='random seed', default=1234, type=int)
    parser.add_argument('--lr', help='learning rate', default=1e-5, type=float)
    parser.add_argument('--optimizer', help='optimizer', default='adamw', type=str)
    parser.add_argument('--scheduler', help='scheduler', default='linear', type=str)
    parser.add_argument('--total-step', help='total training step', default=13000, type=int)
    parser.add_argument('--batch-size-validation',
                        help='batch size for validation (smaller size to save memory)',
                        default=2,
                        type=int)
    parser.add_argument('--warmup-step', help='warmup step (6 percent of total is recommended)', default=700, type=int)
    parser.add_argument('--weight-decay', help='weight decay', default=1e-7, type=float)
    parser.add_argument('--early-stop', help='value of accuracy drop for early stop', default=0.1, type=float)
    parser.add_argument('--inference-mode', help='inference mode', action='store_true')
    parser.add_argument('--fp16', help='fp16', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    opt = get_options()
    classifier = TransformerTokenClassification(
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
        inference_mode=opt.inference_mode,
        early_stop=opt.early_stop,
        fp16=opt.fp16
    )
    if classifier.inference_mode:

        predictions = classifier.predict(['I live in London', '東京は今日も暑いです'])
        print(predictions)

        # while True:
        #     _inp = input('input sentence >>>')
        #     if _inp == 'q':
        #         break
        #     elif _inp == '':
        #         continue
        #     else:
        #         predictions = classifier.predict([_inp])
        #         print(predictions)

    else:
        classifier.train()

