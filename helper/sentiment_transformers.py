""" sentiment analysis model fintuning on hugginface.transformers """
import copy
import traceback
import argparse
import os
import random
import string
import json
import logging
import shutil
import transformers
import torchtext
import torch
import numpy as np
from torch import optim
from torch import nn
from torch.autograd import detect_anomaly
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import util_hf_optimizer

LOGGER = logging.getLogger()
NUM_WORKER = int(os.getenv("NUM_WORKER", '4'))
PROGRESS_INTERVAL = int(os.getenv("PROGRESS_INTERVAL", '500'))
CACHE_DIR = os.getenv("CACHE_DIR", './cache')
CKPT_DIR = os.getenv("CKPT_DIR", './ckpt')
VALID_TRANSFORMER_SEQUENCE_CLASSIFICATION = {
    'xlm-roberta-large': transformers.XLMRobertaForSequenceClassification,
    'bert-base-multilingual-cased': transformers.BertForSequenceClassification}
VALID_TOKENIZER = {
    'xlm-roberta-large': transformers.XLMRobertaTokenizer,
    'bert-base-multilingual-cased': transformers.BertTokenizer}


def get_dataset(data_name: str = 'sst'):
    """ download dataset file and return dictionary including training/validation split """

    def decode_data(iterator, file_prefix, label_dict: dict):
        if os.path.exists(file_prefix + '.text') and os.path.exists(file_prefix + '.label'):
            list_of_text = open(file_prefix + '.text', 'r').read().split('\n')
            list_of_label = [int(l) for l in open(file_prefix + '.label', 'r').read().split('\n')]
            assert len(list_of_label) == len(list_of_text)
            return label_dict, (list_of_text, list_of_label)

        list_text = []
        list_label = []
        for i in iterator:
            if i.label not in label_dict.keys():
                label_dict[i.label] = len(label_dict) + 1
            list_text.append(' '.join(i.text))
            list_label.append(str(label_dict[i.label]))

        with open(file_prefix + '.text', 'w') as f_writer:
            f_writer.write('\n'.join(list_text))

        with open(file_prefix + '.label', 'w') as f_writer:
            f_writer.write('\n'.join(list_label))

        return label_dict, None

    data_field, label_field = torchtext.data.Field(sequential=True), torchtext.data.Field(sequential=False)
    if data_name == 'imdb':
        iterator_split = torchtext.datasets.IMDB.splits(data_field, root=CACHE_DIR, label_field=label_field)
    elif data_name == 'sst':
        iterator_split = torchtext.datasets.SST.splits(data_field, root=CACHE_DIR, label_field=label_field)
    else:
        raise ValueError('unknown dataset: %s' % data_name)

    data_split, data = list(), None
    if os.path.exists(os.path.join(CACHE_DIR, data_name, 'label.json')):
        label_dictionary = json.load(open(os.path.join(CACHE_DIR, data_name, 'label.json')))
    else:
        label_dictionary = dict()

    for name, it in zip(['train', 'valid', 'test'], iterator_split):
        _file_prefix = os.path.join(CACHE_DIR, data_name, name)
        label_dictionary, data = decode_data(it, file_prefix=_file_prefix, label_dict=label_dictionary)
        if data is None:
            _, data = decode_data(it, file_prefix=_file_prefix, label_dict=label_dictionary)
        data_split.append(data)

    with open(os.path.join(CACHE_DIR, data_name, 'label.json'), 'w') as f:
        json.dump(label_dictionary, f)
    num_labels = len(list(label_dictionary.keys()))
    return data_split, num_labels


class Dataset(torch.utils.data.Dataset):
    """ torch.utils.data.Dataset instance """

    def __init__(self,
                 data: list,
                 label: list,
                 transformer: str,
                 device: str = 'cpu'):
        self.data = data
        self.label = [int(l) for l in label]
        self.device = device
        # setup tokenizer as transformation of input data
        self.tokenizer = VALID_TOKENIZER[transformer].from_pretrained(transformer, cache_dir=CACHE_DIR)

    def encode(self, text):
        token_ids = self.tokenizer.encode(text)
        if self.tokenizer.max_len <= len(token_ids):
            token_ids = token_ids[:self.tokenizer.max_len]
        else:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (self.tokenizer.max_len - len(token_ids))
        return token_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out_data = torch.tensor(self.encode(self.data[idx]), dtype=torch.long).to(self.device)
        out_label = torch.tensor(self.label[idx], dtype=torch.long).to(self.device)
        return out_data, out_label


class ParameterManager:
    """ Parameter manager for model training """

    def __init__(self,
                 prefix: str = None,
                 checkpoint: str = None,
                 **kwargs):

        """ Parameter manager for model training

         Parameter
        -------------------
        prefix: prefix to filename
        checkpoint: existing checkpoint name if you want to load
        kwargs: model parameters
        """
        self.prefix = prefix
        self.checkpoint_dir, self.parameter = self.__versioning(kwargs, checkpoint)
        LOGGER.debug('checkpoint: %s' % self.checkpoint_dir)

    def __call__(self, key):
        """ retrieve a parameter """
        if key not in self.parameter.keys():
            raise ValueError('unknown parameter %s' % key)
        return self.parameter[key]

    @staticmethod
    def random_string(string_length=10, exceptions: list = None):
        """ Generate a random string of fixed length """
        while True:
            letters = string.ascii_lowercase
            random_letters = ''.join(random.choice(letters) for i in range(string_length))
            if exceptions is None or random_letters not in exceptions:
                break
        return random_letters

    def __versioning(self, parameter: dict = None, checkpoint: str = None):
        """ Checkpoint versioner: Either of `config` or `checkpoint` need to be specified (`config` has priority)

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
            LOGGER.debug('issue new checkpoint id')
            # check if there are any checkpoints with same hyperparameters
            version_name = []
            for parameter_path in glob(os.path.join(CKPT_DIR, '*/hyperparameters.json')):
                _dir = parameter_path.replace('/hyperparameters.json', '')
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

            new_checkpoint = self.random_string(exceptions=version_name)
            if self.prefix:
                new_checkpoint = '_'.join([self.prefix, new_checkpoint])
            new_checkpoint_path = os.path.join(CKPT_DIR, new_checkpoint)
            os.makedirs(new_checkpoint_path, exist_ok=True)
            with open(os.path.join(new_checkpoint_path, 'hyperparameters.json'), 'w') as _f:
                json.dump(parameter, _f)
            return new_checkpoint_path, parameter

        else:
            LOGGER.debug('load existing checkpoint')
            checkpoints = glob(os.path.join(CKPT_DIR, checkpoint, 'hyperparameters.json'))
            if len(checkpoints) >= 2:
                raise ValueError('Checkpoints are duplicated: %s' % str(checkpoints))
            elif len(checkpoints) == 0:
                raise ValueError('No checkpoint: %s' % os.path.join(CKPT_DIR, checkpoint))
            else:
                parameter = json.load(open(checkpoints[0]))
                target_checkpoints_path = checkpoints[0].replace('/hyperparameters.json', '')
                return target_checkpoints_path, parameter


class TransformerSequenceClassifier:

    def __init__(self,
                 dataset: str,
                 # load_best_model: bool = False,
                 checkpoint: str=None,
                 **kwargs):
        LOGGER.debug('*** initialize network ***')

        # checkpoint versioning
        self.param = ParameterManager(prefix=dataset, checkpoint=checkpoint, dataset=dataset, **kwargs)
        self.checkpoint_model = os.path.join(self.param.checkpoint_dir, 'model.pt')

        # fix random seed
        random.seed(self.param('random_seed'))
        np.random.seed(self.param('random_seed'))
        torch.manual_seed(self.param('random_seed'))

        # model setup
        _, num_labels = get_dataset(self.param('dataset'))
        model_seq_cls_class = VALID_TRANSFORMER_SEQUENCE_CLASSIFICATION[self.param('transformer')]
        self.model_seq_cls = model_seq_cls_class.from_pretrained(
            self.param('transformer'), cache_dir=CACHE_DIR, num_labels=num_labels)

        # GPU allocation
        self.n_gpu = torch.cuda.device_count()
        if self.n_gpu == 1:
            self.model_seq_cls = self.model_seq_cls.cuda()
        elif self.n_gpu > 1:
            self.model_seq_cls = torch.nn.DataParallel(self.model_seq_cls.cuda())
        else:
            self.n_gpu = 0
        LOGGER.debug('running on %i GPUs' % self.n_gpu)
        self.device = 'cuda' if self.n_gpu > 0 else 'cpu'

        # optimizer
        if self.param("optimizer") == 'adamw':
            self.optimizer = util_hf_optimizer.AdamW(
                self.model_seq_cls.parameters(), lr=self.param('lr'), weight_decay=self.param('weight_decay'))
        elif self.param("optimizer") == 'adam':
            self.optimizer = optim.Adam(
                self.model_seq_cls.parameters(), lr=self.param('lr'), weight_decay=self.param('weight_decay'))
        elif self.param("optimizer") == 'sgd':
            self.optimizer = optim.SGD(
                self.model_seq_cls.parameters(), lr=self.param('lr'), weight_decay=self.param('weight_decay'))
        else:
            raise ValueError('bad optimizer: %s' % self.param("optimizer"))

        # scheduler
        if self.param('scheduler') == 'constant':
            self.scheduler = util_hf_optimizer.get_constant_schedule(self.optimizer)
        elif self.param('scheduler') == 'linear':
            self.scheduler = util_hf_optimizer.get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.param('warmup_step'),
                num_training_steps=self.param('total_step'))
        else:
            raise ValueError('bad scheduler: %s' % self.param('scheduler'))

        # load ckpt
        if os.path.exists(self.checkpoint_model):
            ckpt = torch.load(self.checkpoint_model, map_location=self.device)
            # if load_best_model:
            #     self.model_seq_cls.load_state_dict(ckpt['best_model_state_dict'])
            # else:
            self.model_seq_cls.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            self.__step = ckpt['step']  # num of training step
            self.__epoch = ckpt['epoch']  # num of epoch
            self.__best_val_loss = ckpt['best_val_loss']
            self.__best_val_loss_step = ckpt['best_val_loss_step']
            self.__best_model_wts = ckpt['best_model_state_dict']
            LOGGER.debug('load ckpt from %s' % self.checkpoint_model)
        else:
            self.__step = 0
            self.__epoch = 0
            self.__best_val_loss_ = None
            self.__best_val_loss_step = None
            self.__best_model_wts = None

        # log
        self.writer = SummaryWriter(log_dir=self.param.checkpoint_dir)
        self.show_config()

    def show_config(self):
        model_size = 0
        for k, v in self.model_seq_cls.__dict__['_modules'].items():
            if hasattr(v, 'weight'):
                model_size += np.prod(v.weight.shape)
                LOGGER.debug(' - [weight size] %s: %s' % (k, str(list(v.weight.shape))))
        LOGGER.debug(' - %i variables in total' % model_size)
        for k, v in self.param.parameter.items():
            LOGGER.debug(' - [param] %s: %s' % (k, str(v)))

    def predict(self, x):
        pass
        # outputs = self.model_seq_cls(inputs, outputs)
        # loss, logit = outputs[0:2]

    def train(self):

        # setup data loader
        dataset_split, _ = get_dataset(self.param('dataset'))
        data_loader_train = torch.utils.data.DataLoader(
            Dataset(*dataset_split[0], transformer=self.param('transformer'), device=self.device),
            batch_size=self.param('batch_size'),
            shuffle=True,
            num_workers=NUM_WORKER,
            drop_last=True)
        data_loader_valid = torch.utils.data.DataLoader(
            Dataset(*dataset_split[1], transformer=self.param('transformer'), device=self.device),
            batch_size=self.param('batch_size'))
        if len(dataset_split) > 2:
            data_loader_test = torch.utils.data.DataLoader(
                Dataset(*dataset_split[1], transformer=self.param('transformer'), device=self.device),
                batch_size=self.param('batch_size'))
        else:
            data_loader_test = None

        try:
            with detect_anomaly():
                while True:
                    if_training_finish = self.__epoch_train(data_loader_train)
                    if_early_stop = self.__epoch_valid(data_loader_valid, prefix='valid')
                    if if_training_finish or if_early_stop:
                        if data_loader_test:
                            self.__epoch_valid(data_loader_valid, prefix='test')
                        break
                    self.__epoch += 1

        except RuntimeError:
            LOGGER.debug(traceback.format_exc())
            LOGGER.info('*** RuntimeError (NaN found, see above log in detail) ***')

        except KeyboardInterrupt:
            LOGGER.info('*** KeyboardInterrupt ***')

        if self.__best_val_loss is None:
            exit('nothing to be saved')

        LOGGER.debug('[training completed] best model: valid loss %0.3f at step %i'
                     % (self.__best_val_loss, self.__best_val_loss_step))
        torch.save({
            'model_state_dict': self.model_seq_cls.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.__step,
            'epoch': self.__epoch,
            'best_val_loss': self.__best_val_loss,
            'best_val_loss_step': self.__best_val_loss_step,
            'best_model_state_dict': self.__best_model_wts
        }, self.checkpoint_model)
        self.writer.close()
        LOGGER.info('ckpt saved at %s' % self.checkpoint_model)

    def __epoch_train(self, data_loader):
        """ train on single epoc, return flag which is True if training has been completed """
        self.model_seq_cls.train()

        for i, (inputs, outputs) in enumerate(data_loader, 1):

            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward: output prediction and get loss
            outputs = self.model_seq_cls(inputs, outputs)
            loss, logit = outputs[0:2]
            _, pred = torch.max(logit, 1)
            # backward: calculate gradient
            loss.backward()
            # gradient clip
            if self.param('clip') is not None:
                nn.utils.clip_grad_norm_(self.model_seq_cls.parameters(), self.param('clip'))
            # optimizer and scheuler step
            self.optimizer.step()
            self.scheduler.step()
            # instantaneous accuracy, loss, and learning rate
            inst_accuracy = ((pred == outputs).cpu().float().mean()).item()
            inst_loss = loss.cpu().item()
            inst_lr = self.optimizer.param_groups[0]['lr']
            # log
            self.writer.add_scalar('train/loss', inst_loss(), self.__step)
            self.writer.add_scalar('train/accuracy', inst_accuracy, self.__step)
            self.writer.add_scalar('learning_rate', inst_lr, self.__step)

            if self.__step % PROGRESS_INTERVAL == 0:
                LOGGER.debug(' * (step %i) accuracy: %.3f, loss: %.3f, lr: %0.6f'
                             % (self.__step, inst_accuracy, inst_loss, inst_lr))

            self.__step += 1

            if self.__step >= self.param('total_step'):
                LOGGER.debug('reached total step')
                return True

        return False

    def __epoch_valid(self, data_loader, prefix: str='valid'):
        """ validation/test """
        self.model_seq_cls.eval()
        accuracy, loss = [], []

        for inputs, outputs in data_loader:
            outputs = self.model_seq_cls(inputs, outputs)
            loss, logit = outputs[0:2]
            _, pred = torch.max(logit, 1)
            accuracy.append(((pred == outputs).cpu().float().mean()).item())
            loss.append(loss.cpu().item())

        accuracy, loss = float(np.mean(accuracy)), float(np.mean(loss))

        self.writer.add_scalar('%s/accuracy' % prefix, accuracy, self.__epoch)
        self.writer.add_scalar('%s/loss' % prefix, loss, self.__epoch)
        LOGGER.debug('[epoch %i] (%s) accuracy: %.3f, loss: %.3f' % (self.__epoch, prefix, accuracy, loss))

        if self.__best_val_loss is None or loss < self.__best_val_loss:
            self.__best_val_loss = loss
            self.__best_val_loss_step = self.__step
            self.__best_model_wts = copy.deepcopy(self.model_seq_cls.state_dict())
        else:
            loss_margin = loss - self.__best_val_loss
            if self.param('tolerance') is not None and self.param('tolerance') < loss_margin:
                LOGGER.debug('early stop as loss exceeds tolerance: %0.2f < %0.2f '
                             % (self.param('tolerance'), loss_margin))
                return True
        return False


def get_options():
    parser = argparse.ArgumentParser(
        description='finetune transformers to sentiment analysis',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--data', help='data (imdb/sst)', default='sst', type=str)
    parser.add_argument('--transformer', help='language model', default='xlm-roberta-large', type=str)
    parser.add_argument('--random-seed', help='random seed', default=1234, type=int)
    parser.add_argument('--lr', help='learning rate', default=0.2, type=float)
    parser.add_argument('--clip', help='gradient clip', default=None, type=float)
    parser.add_argument('--optimizer', help='optimizer', default='adamw', type=str)
    parser.add_argument('--scheduler', help='scheduler', default='linear', type=str)
    parser.add_argument('--total-step', help='total training step', default=100000, type=int)
    parser.add_argument('--warmup-step', help='warmup step', default=5000, type=int)
    parser.add_argument('--weight-decay', help='weight decay', default=1e-7, type=float)
    parser.add_argument('--tolerance', help='tolerance for valid loss', default=None, type=float)
    parser.add_argument('--checkpoint', help='checkpoint to load', default=None, type=str)
    parser.add_argument('--inference-mode', help='inference mode', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    opt = get_options()
    classifier = TransformerSequenceClassifier(
        checkpoint=opt.checkpoint,
        dataset=opt.data,
        transformer=opt.transformer,
        random_seed=opt.random_seed,
        lr=opt.lr,
        clip=opt.clip,
        optimizer=opt.optimizer,
        scheduler=opt.scheduler,
        total_step=opt.total_step,
        warmup_step=opt.warmup_step,
        tolerance=opt.tolerance,
        weight_decay=opt.weight_decay
    )
    if opt.inference_mode:
        classifier.train()
    else:
        classifier.predict()

