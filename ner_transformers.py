""" self-contained NER finetuning on hugginface.transformers """

import argparse
import os
import random
import json
import logging
import re
from time import time
from logging.config import dictConfig
from itertools import chain

import transformers
import torch
from torch import optim
from torch import nn
from torch.autograd import detect_anomaly
from torch.utils.tensorboard import SummaryWriter
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report, accuracy_score

from get_dataset import get_dataset_ner
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
PAD_TOKEN_LABEL_ID = nn.CrossEntropyLoss().ignore_index


class Transforms:
    """ Text encoder with transformers tokenizer """

    def __init__(self,
                 transformer_tokenizer: str,
                 max_seq_length: int = None,
                 pad_to_max_length: bool = True,
                 language: str = 'en'):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(transformer_tokenizer, cache_dir=CACHE_DIR)
        if max_seq_length and max_seq_length > self.tokenizer.max_len:
            raise ValueError('`max_seq_length should be less than %i' % self.tokenizer.max_len)
        self.max_seq_length = max_seq_length if max_seq_length else self.tokenizer.max_len
        self.pad_to_max_length = pad_to_max_length
        self.language = language
        self.pad_ids = {"labels": PAD_TOKEN_LABEL_ID, "input_ids": self.tokenizer.pad_token_id, "__default__": 0}
        # find tokenizer-depend prefix
        sentence_go_around = ''.join(self.tokenizer.tokenize('get tokenizer specific prefix'))
        self.prefix = sentence_go_around[:list(re.finditer('get', sentence_go_around))[0].span()[0]]
        # find special tokens to be added
        self.sp_token_start, self.sp_token_end = self.get_add_token()

    def get_add_token(self):
        encode_first = self.tokenizer.encode_plus('test')
        inner_token_ids = []
        sp_token_start = {k: [] for k in encode_first.keys()}
        sp_token_end = {k: [] for k in encode_first.keys()}
        for n, i in enumerate(encode_first['input_ids']):
            if i in self.tokenizer.all_special_ids:
                for k, v in encode_first.items():
                    if len(inner_token_ids) == 0:
                        sp_token_start[k] += [v[n]]
                    else:
                        sp_token_end[k] += [v[n]]
            else:
                inner_token_ids += [i]
        sp_token_end['labels'] = [self.pad_ids['labels']] * len(sp_token_end['input_ids'])
        sp_token_start['labels'] = [self.pad_ids['labels']] * len(sp_token_start['input_ids'])
        return sp_token_start, sp_token_end

    def fixed_encode_en(self, tokens, labels: list = None):
        """ fixed encoding for language with halfspace in between words """
        encode = self.tokenizer.encode_plus(
            ' '.join(tokens), max_length=self.max_seq_length, pad_to_max_length=self.pad_to_max_length, truncation=self.pad_to_max_length)
        if labels:
            assert len(tokens) == len(labels)
            fixed_labels = list(chain(*[
                [label] + [self.pad_ids['labels']] * (len(self.tokenizer.tokenize(word)) - 1)
                for label, word in zip(labels, tokens)]))
            fixed_labels = [self.pad_ids['labels']] * len(self.sp_token_start['labels']) + fixed_labels
            fixed_labels = fixed_labels[:min(len(fixed_labels), self.max_seq_length - len(self.sp_token_end['labels']))]
            fixed_labels = fixed_labels + [self.pad_ids['labels']] * (self.max_seq_length - len(fixed_labels))
            encode['labels'] = fixed_labels
        return encode

    def fixed_encode_ja(self, tokens, labels: list = None, dummy: str = '@'):
        """ fixed encoding for language without halfspace in between words """
        # get special tokens at start/end of sentence based on first token
        encode_all = self.tokenizer.batch_encode_plus(tokens)
        # token_ids without prefix/special tokens
        # `wifi` will be treated as `_wifi` and change the tokenize result, so add dummy on top of the sentence to fix
        token_ids_all = [[self.tokenizer.convert_tokens_to_ids(_t.replace(self.prefix, '').replace(dummy, ''))
                          for _t in self.tokenizer.tokenize(dummy+t)
                          if len(_t.replace(self.prefix, '').replace(dummy, '')) > 0]
                         for t in tokens]

        for n in range(len(tokens)):
            if n == 0:
                encode = {k: v[n][:-len(self.sp_token_end[k])] for k, v in encode_all.items()}
                if labels:
                    encode['labels'] = [self.pad_ids['labels']] * len(self.sp_token_start['labels']) + [labels[n]]
                    encode['labels'] += [self.pad_ids['labels']] * (len(encode['input_ids']) - len(encode['labels']))
            else:
                encode['input_ids'] += token_ids_all[n]
                # other attribution without prefix/special tokens
                tmp_encode = {k: v[n] for k, v in encode_all.items()}
                input_ids_with_prefix = \
                    tmp_encode.pop('input_ids')[len(self.sp_token_start['input_ids']):-len(self.sp_token_end['input_ids'])]
                prefix_length = len(input_ids_with_prefix) - len(token_ids_all[n])
                for k, v in tmp_encode.items():
                    encode[k] += v[len(self.sp_token_start['input_ids']) + prefix_length:-len(self.sp_token_end['input_ids'])]
                if labels:
                    encode['labels'] += [labels[n]] + [self.pad_ids['labels']] * (len((token_ids_all[n])) - 1)

        # add special token at the end and padding/truncate accordingly
        for k in encode.keys():
            encode[k] = encode[k][:min(len(encode[k]), self.max_seq_length - len(self.sp_token_end[k]))]
            encode[k] += self.sp_token_end[k]
            pad_id = self.pad_ids[k] if k in self.pad_ids.keys() else self.pad_ids['__default__']
            encode[k] += [pad_id] * (self.max_seq_length - len(encode[k]))
        return encode

    def __call__(self, text, labels: list = None):
        if type(text) is str:
            return self.tokenizer.encode_plus(
                text, max_length=self.max_seq_length, pad_to_max_length=self.pad_to_max_length, truncation=self.pad_to_max_length)
        elif type(text) is list:
            if self.language == 'en':
                return self.fixed_encode_en(text, labels)
            elif self.language == 'ja':
                return self.fixed_encode_ja(text, labels)
            else:
                raise ValueError('unknown language: {}'.format(self.language))

    @property
    def all_special_ids(self):
        return self.tokenizer.all_special_ids

    @property
    def all_special_tokens(self):
        return self.tokenizer.all_special_tokens

    def tokenize(self, *args, **kwargs):
        return self.tokenizer.tokenize(*args, **kwargs)


class Dataset(torch.utils.data.Dataset):
    """ torch.utils.data.Dataset with transformer tokenizer """

    def __init__(self, data: list, transform_function, label: list = None):
        self.data = data  # list of half-space split tokens
        self.transform_function = transform_function
        self.label = label if label else [None] * len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encode = self.transform_function(self.data[idx], self.label[idx])
        float_list = ['attention_mask']
        encode = {k: torch.tensor(v, dtype=torch.float32) if k in float_list else torch.tensor(v, dtype=torch.long)
                  for k, v in encode.items()}
        return encode


class TransformerTokenClassification:
    """ finetune transformers on token classification """

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
            self.dataset_split, self.label_to_id = get_dataset_ner(self.args.dataset)
            with open(os.path.join(self.args.checkpoint_dir, 'label_to_id.json'), 'w') as f:
                json.dump(self.label_to_id, f)
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
        self.transforms = Transforms(self.args.transformer, self.args.max_seq_length, language=self.args.language)

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
            return None

    def predict(self, x: list):
        """ model inference """
        self.model.eval()
        data_loader = torch.utils.data.DataLoader(
            Dataset(x, transform_function=self.transforms), num_workers=NUM_WORKER, batch_size=len(x))
        prediction = []
        for encode in data_loader:
            logit = self.model(**{k: v.to(self.device) for k, v in encode.items()})[0]
            _, _pred = torch.max(logit, dim=-1)
            prediction += [[self.id_to_label[_p] for _p in batch] for batch in _pred.cpu().tolist()]
        return prediction

    def test(self):
        LOGGER.addHandler(logging.FileHandler(os.path.join(self.args.checkpoint_dir, 'logger_test.log')))
        writer = SummaryWriter(log_dir=self.args.checkpoint_dir)
        if self.dataset_split is None:
            self.dataset_split = get_dataset_ner(self.args.dataset, label_to_id=self.label_to_id, allow_update=False)
        data_loader_test = {k: torch.utils.data.DataLoader(
            Dataset(**v, transform_function=self.transforms),
            num_workers=NUM_WORKER,
            batch_size=self.args.batch_size)
            for k, v in self.dataset_split.items() if k not in ['train', 'valid']}
        LOGGER.info('data_loader_test: %s' % str(list(data_loader_test.keys())))
        assert len(data_loader_test.keys()) != 0, 'no test set found'
        start_time = time()
        for k, v in data_loader_test.items():
            self.__epoch_valid(v, writer=writer, prefix=k)
            self.release_cache()
        writer.close()
        LOGGER.info('[test completed, %0.2f sec in total]' % (time() - start_time))

    def train(self):
        LOGGER.addHandler(logging.FileHandler(os.path.join(self.args.checkpoint_dir, 'logger_train.log')))
        if self.dataset_split is None:
            self.dataset_split = get_dataset_ner(self.args.dataset, label_to_id=self.label_to_id, allow_update=False)
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

    def __epoch_train(self, data_loader, writer):
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
            writer.add_scalar('train/loss', inst_loss, self.__step)
            writer.add_scalar('train/learning_rate', inst_lr, self.__step)
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
        try:
            LOGGER.info('[epoch %i] (%s) \n %s' % (self.__epoch, prefix, classification_report(seq_true, seq_pred)))
        except ZeroDivisionError:
            LOGGER.info('[epoch %i] (%s) * classification_report raises `ZeroDivisionError`' % (self.__epoch, prefix))
        writer.add_scalar('%s/f1' % prefix, f1_score(seq_true, seq_pred), self.__epoch)
        writer.add_scalar('%s/recall' % prefix, recall_score(seq_true, seq_pred), self.__epoch)
        writer.add_scalar('%s/precision' % prefix, precision_score(seq_true, seq_pred), self.__epoch)
        writer.add_scalar('%s/accuracy' % prefix, accuracy_score(seq_true, seq_pred), self.__epoch)
        writer.add_scalar('%s/loss' % prefix, float(sum(list_loss) / len(list_loss)), self.__epoch)
        if prefix == 'valid':
            score = f1_score(seq_true, seq_pred)
            if self.__best_val_score is None or score > self.__best_val_score:
                self.__best_val_score = score
            if self.args.early_stop and self.__best_val_score - score > self.args.early_stop:
                return True
        return False

    def release_cache(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()


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
    parser.add_argument('--test', help='run over testdataset', action='store_true')
    parser.add_argument('--fp16', help='fp16', action='store_true')
    parser.add_argument('-l', '--language', help='language', default='en', type=str)
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
        early_stop=opt.early_stop,
        fp16=opt.fp16,
        language=opt.language
    )
    if opt.test_sentence is not None:
        # test_sentence = opt.test_sentence.split(',')
        # predictions = classifier.predict(['I live in London', '東京は今日も暑いです'])
        # print(predictions)

        while True:
            _inp = input('input sentence >>>')
            if _inp == 'q':
                break
            elif _inp == '':
                continue
            else:
                predictions = classifier.predict([_inp])
                print(predictions)

    else:
        if opt.test:
            classifier.test()
        else:
            classifier.train()

