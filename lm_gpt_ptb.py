""" pytorch GPT implementation, train on PTB

reference
- official codebase (tf) https://github.com/openai/gpt-2
- huggingface (torch) https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_gpt2.py#L329
- fairseq (torch)
    model https://github.com/pytorch/fairseq/blob/master/fairseq/models/transformer_lm.py
    transformer https://github.com/pytorch/fairseq/blob/master/fairseq/models/transformer.py

Optimizer!!!
https://github.com/huggingface/transformers/blob/master/examples/run_lm_finetuning.py
https://github.com/huggingface/transformers/blob/90b7df444fc30d5f476e5ab32d1f89340998a28d/src/transformers/optimization.py#L96
"""

# for model
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# for parameter manager
import json
import os
import shutil
import random
import string
import toml
from glob import glob

# for logger
import logging
from logging.config import dictConfig

from transformer_module import BaseGPT2


def create_log():
    """ simple Logger
    Usage
    -------------------
    logger.info(message)
    logger.error(error)
    """
    logging_config = dict(
        version=1,
        formatters={
            'f': {'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}
        },
        handlers={
            'h': {'class': 'logging.StreamHandler',
                  'formatter': 'f',
                  'level': logging.DEBUG}
        },
        root={
            'handlers': ['h'],
            'level': logging.DEBUG,
        },
    )
    dictConfig(logging_config)
    logger = logging.getLogger()
    return logger


class ParameterManager:
    """ Parameter manager for model training """

    def __init__(self,
                 checkpoint: str = None,
                 checkpoint_dir: str = None,
                 default_parameter: str = None,
                 **kwargs):

        """ Parameter manager for model training
        - loading model: {checkpoint_dir}/{checkpoint}
        - new model: {checkpoint_dir}/new_unique_checkpoint_id

         Parameter
        -------------------
        checkpoint: existing checkpoint name if you want to load
        checkpoint_dir: checkpoint dir
        default_parameter: path to toml file containing default parameters
        kwargs: model parameters
        """
        self.__logger = create_log()
        if checkpoint_dir is None:
            checkpoint_dir = os.path.dirname(os.path.abspath(__file__))
        self.__logger.debug('checkpoint_dir: %s' % checkpoint_dir)
        if default_parameter is None:
            self.__logger.debug('no default parameter')
            parameter = kwargs
        else:
            self.__logger.debug('fetch default parameter from: %s' % default_parameter)
            parameter = dict()
            default_parameter = self.get_default_parameter(default_parameter)

            for k, v in default_parameter.items():
                if k in kwargs.keys():
                    parameter[k] = kwargs[k]
                else:
                    parameter[k] = v

        self.checkpoint_dir, self.parameter = self.__versioning(checkpoint_dir, parameter, checkpoint)
        self.__logger.debug('checkpoint: %s' % self.checkpoint_dir)

    def __call__(self, key):
        """ retrieve a parameter """
        if key not in self.parameter.keys():
            raise ValueError('unknown parameter %s' % key)
        return self.parameter[key]

    @staticmethod
    def get_default_parameter(default_parameter_toml_file: str):
        """ Get default parameter from toml file """
        assert default_parameter_toml_file.endswith('.toml')
        if not os.path.exists(default_parameter_toml_file):
            raise ValueError('no toml file: %s' % default_parameter_toml_file)
        parameter = toml.load(open(default_parameter_toml_file, "r"))
        parameter = dict([(k, v if v != '' else None) for k, v in parameter.items()])  # '' -> None
        return parameter

    @staticmethod
    def random_string(string_length=10, exceptions: list = None):
        """ Generate a random string of fixed length """
        if exceptions is None:
            exceptions = []
        while True:
            letters = string.ascii_lowercase
            random_letters = ''.join(random.choice(letters) for i in range(string_length))
            if random_letters not in exceptions:
                break
        return random_letters

    def __versioning(self, checkpoint_dir: str, parameter: dict = None, checkpoint: str = None):
        """ Checkpoint versioner: Either of `config` or `checkpoint` need to be specified (`config` has priority)

         Parameter
        ---------------------
        checkpoint_dir: directory where checkpoints will be saved, eg) `checkpoints/bert`
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
            self.__logger.debug('issue new checkpoint id')
            # check if there are any checkpoints with same hyperparameters
            version_name = []
            for parameter_path in glob(os.path.join(checkpoint_dir, '*/hyperparameters.json')):
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
            new_checkpoint_path = os.path.join(checkpoint_dir, new_checkpoint)
            os.makedirs(new_checkpoint_path, exist_ok=True)
            with open(os.path.join(new_checkpoint_path, 'hyperparameters.json'), 'w') as _f:
                json.dump(parameter, _f)
            return new_checkpoint_path, parameter

        else:
            self.__logger.debug('load existing checkpoint')
            checkpoints = glob(os.path.join(checkpoint_dir, checkpoint, 'hyperparameters.json'))
            if len(checkpoints) >= 2:
                raise ValueError('Checkpoints are duplicated: %s' % str(checkpoints))
            elif len(checkpoints) == 0:
                raise ValueError('No checkpoint: %s' % os.path.join(checkpoint_dir, checkpoint))
            else:
                parameter = json.load(open(checkpoints[0]))
                target_checkpoints_path = checkpoints[0].replace('/hyperparameters.json', '')
                return target_checkpoints_path, parameter


class BatchFeeder:
    """ Pytorch batch feeding iterator for language model training """

    def __init__(self,
                 batch_size,
                 num_steps,
                 sequence):
        """ Pytorch batch feeding iterator for language model training

         Parameter
        -------------------
        batch_size: int
            batch size
        num_steps: int
            sequence truncation size
        sequence: list
            integer token id sequence to feed
        """
        self._index = 0
        self.batch_size = batch_size
        self.num_steps = num_steps
        seq = torch.LongTensor(sequence)
        self.data_size = seq.size(0)

        n_batch = self.data_size // self.batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        seq = seq.narrow(0, 0, n_batch * self.batch_size)
        # Evenly divide the data across the bsz batches.
        self._data = seq.view(self.batch_size, -1).t().contiguous()
        if torch.cuda.device_count() >= 1:
            self._data = self._data.cuda()

    def __len__(self):
        return (self.data_size // self.batch_size - 1) // self.num_steps

    def __iter__(self):
        return self

    def __next__(self):
        """ next batch for train data (size is `self._batch_size`) loop for self._iteration_number

         Return
        -----------------
        (inputs, outputs): list (batch_size, num_steps)
        """
        if (self._index + 1) * self.num_steps + 1 > self._data.size(0):
            self._index = 0
            raise StopIteration
        x = self._data[self._index * self.num_steps:(self._index + 1) * self.num_steps, :]
        y = self._data[self._index * self.num_steps + 1:(self._index + 1) * self.num_steps + 1, :]
        self._index += 1
        return x, y


class GPT2:
    """ GPT2 language model """

    def __init__(self,
                 model_version: str = 'small',
                 checkpoint_dir: str = None,
                 **kwargs):
        """ GPT2 language model """
        self.__logger = create_log()
        self.__logger.debug('initialize network: \n*** GPT2 ***\n')
        # setup parameter
        self.__param = ParameterManager(
            checkpoint_dir=checkpoint_dir,
            default_parameter='./parameters/lm_gpt_%s_ptb.toml' % model_version,
            **kwargs)
        self.__checkpoint_model = os.path.join(self.__param.checkpoint_dir, 'model.pt')
        # build network
        self.__net = BaseGPT2(
            n_layer=self.__param("n_layer"),
            n_embedding=self.__param("n_embedding"),
            n_state_ffn=self.__param("n_state_ffn"),
            n_head=self.__param("n_head"),
            n_context=self.__param("n_context"),
            max_cache_size=self.__param("max_cache_size"),
            residual_dropout=self.__param("residual_dropout"),
            attention_dropout=self.__param("attention_dropout"),
            embedding_dropout=self.__param("embedding_dropout"),
            vocab_size=self.__param("vocab_size"),
        )
        # GPU allocation
        if torch.cuda.device_count() >= 1:
            self.__logger.debug('running on GPU')
            self.if_use_gpu = True
            self.__net = self.__net.cuda()
        else:
            self.if_use_gpu = False
        # optimizer
        if self.__param('optimizer') == 'adam':
            self.__optimizer = optim.Adam(
                self.__net.parameters(), lr=self.__param('lr'), weight_decay=self.__param('weight_decay'))
        elif self.__param('optimizer') == 'sgd':
            self.__optimizer = optim.SGD(
                self.__net.parameters(), lr=self.__param('lr'), weight_decay=self.__param('weight_decay'))
        else:
            raise ValueError('unknown optimizer %s' % self.__param('optimizer'))

        # loss definition (CrossEntropyLoss includes softmax inside)
        self.__loss = nn.CrossEntropyLoss()

        # load pre-trained ckpt
        if os.path.exists(self.__checkpoint_model):
            self.__net.load_state_dict(torch.load(self.__checkpoint_model))
            self.__logger.debug('load ckpt from %s' % self.__checkpoint_model)
        # log
        self.__writer = SummaryWriter(log_dir=self.__param.checkpoint_dir)
        self.__sanity_check()

    @property
    def hyperparameters(self):
        return self.__param

    def __sanity_check(self):
        """ sanity check as logging model size """
        self.__logger.debug('trainable variables')
        model_size = 0
        for name, param in self.__net.named_parameters():
            if param.requires_grad:
                __shape = list(param.data.shape)
                model_size += np.prod(__shape)
                self.__logger.debug(' - [weight size] %s: %s' % (name, str(__shape)))
        self.__logger.debug(' - %i variables in total' % model_size)
        self.__logger.debug('hyperparameters')
        for k, v in self.__param.parameter.items():
            self.__logger.debug(' - [param] %s: %s' % (k, str(v)))

    def train(self,
              epoch: int,
              data_train: list,
              data_valid: list,
              data_test: list=None,
              progress_interval: int = 100):
        """ train model """
        best_model_wts = copy.deepcopy(self.__net.state_dict())
        best_ppl = 100000
        best_epoch = 0

        self.__logger.debug('initialize batch feeder')
        loader_train = BatchFeeder(
            batch_size=self.__param('batch_size'), num_steps=self.__param('sequence_length'), sequence=data_train)
        loader_valid = BatchFeeder(
            batch_size=self.__param('batch_size'), num_steps=self.__param('sequence_length'), sequence=data_valid)
        if data_test:
            loader_test = BatchFeeder(
                batch_size=self.__param('batch_size'), num_steps=self.__param('sequence_length'), sequence=data_test)
        else:
            loader_test = None

        try:
            for e in range(epoch):  # loop over the epoch

                loss, ppl = self.__epoch_train(loader_train, epoch_n=e, progress_interval=progress_interval)
                self.__logger.debug('[epoch %i/%i] (train) loss: %.3f, ppl: %.3f' % (e, epoch, loss, ppl))

                loss, ppl = self.__epoch_valid(loader_valid, epoch_n=e)
                self.__logger.debug('[epoch %i/%i] (valid) loss: %.3f, ppl: %.3f' % (e, epoch, loss, ppl))

                if ppl < best_ppl:
                    best_model_wts = copy.deepcopy(self.__net.state_dict())
                    best_epoch = e
                    best_ppl = ppl
            if loader_test:
                loss, ppl = self.__epoch_test(loader_test)
                self.__logger.debug('(test) loss: %.3f, acc: %.3f' % (loss, ppl))
        except KeyboardInterrupt:
            self.__logger.info('*** KeyboardInterrupt ***')

        self.__writer.close()
        self.__logger.debug('best model: epoch %i, valid ppt %0.3f' % (best_epoch, best_ppl))
        torch.save(best_model_wts, self.__checkpoint_model)
        self.__logger.debug('complete training: best model ckpt was saved at %s' % self.__checkpoint_model)

    def __epoch_train(self,
                      data_loader,
                      epoch_n: int,
                      progress_interval: int = 100000):
        """ single epoch process for training """
        self.__net.train()
        perplexity = -100
        hidden_state = None
        full_seq_length = 0
        full_loss = 0

        for i, data in enumerate(data_loader, 1):
            # get the inputs (data is a list of [inputs, labels])
            inputs, outputs = data
            # zero the parameter gradients
            self.__optimizer.zero_grad()
            # forward: output prediction and get loss
            (logit, prob, pred), hidden_state = self.__net(inputs, hidden_state)
            # backward: calculate gradient
            logit = logit.view(-1, logit.size(-1))
            outputs = outputs.view(-1)
            tmp_loss = self.__loss(logit, outputs)
            tmp_loss.backward()
            # gradient clip
            if self.__param('clip') is not None:
                nn.utils.clip_grad_norm_(self.__net.parameters(), self.__param('clip'))
            # optimize
            self.__optimizer.step()
            # log
            tmp_loss = tmp_loss.cpu().item()
            full_loss += len(outputs) * tmp_loss
            full_seq_length += len(outputs)
            perplexity = np.exp(min(30, full_loss / full_seq_length))
            self.__writer.add_scalar('train/loss', tmp_loss, i + epoch_n * len(data_loader))
            self.__writer.add_scalar('train/perplexity', perplexity, i + epoch_n * len(data_loader))

            if i % progress_interval == 0:
                self.__logger.debug(' * (%i/%i) loss: %.3f, ppl: %.3f' % (i, len(data_loader), tmp_loss, perplexity))

        mean_loss = full_loss / full_seq_length
        return mean_loss, perplexity

    def __epoch_valid(self,
                      data_loader,
                      epoch_n: int):
        """ single epoch process for validation """
        self.__net.eval()
        hidden_state = None
        full_seq_length = 0
        full_loss = 0
        for data in data_loader:
            inputs, outputs = data
            (logit, prob, pred), hidden_state = self.__net(inputs, hidden_state)
            logit = logit.view(-1, logit.size(-1))
            outputs = outputs.view(-1)
            tmp_loss = self.__loss(logit, outputs)
            tmp_loss = tmp_loss.cpu().item()
            full_loss += len(outputs) * tmp_loss
            full_seq_length += len(outputs)
        mean_loss = full_loss / full_seq_length
        perplexity = np.exp(min(30, full_loss / full_seq_length))
        self.__writer.add_scalar('valid/perplexity', perplexity, epoch_n)
        self.__writer.add_scalar('valid/loss', mean_loss, epoch_n)
        return mean_loss, perplexity

    def __epoch_test(self, data_loader):
        """ single epoch process for test """
        self.__net.eval()
        hidden_state = None
        full_seq_length = 0
        full_loss = 0
        for data in data_loader:
            inputs, outputs = data
            (logit, prob, pred), hidden_state = self.__net(inputs, hidden_state)
            logit = logit.view(-1, logit.size(-1))
            outputs = outputs.view(-1)
            tmp_loss = self.__loss(logit, outputs)
            tmp_loss = tmp_loss.cpu().item()
            full_loss += len(outputs) * tmp_loss
            full_seq_length += len(outputs)
        mean_loss = full_loss / full_seq_length
        perplexity = np.exp(min(30, full_loss / full_seq_length))
        return mean_loss, perplexity


if __name__ == '__main__':
    with open('./data/penn-treebank/ptb.train.eos.id.txt', 'r') as f:
        _data_train = [int(i) for i in f.read().split()]
    with open('./data/penn-treebank/ptb.valid.eos.id.txt', 'r') as f:
        _data_valid = [int(i) for i in f.read().split()]
    with open('./data/penn-treebank/ptb.test.eos.id.txt', 'r') as f:
        _data_test = [int(i) for i in f.read().split()]

    _model = GPT2(checkpoint_dir='./ckpt/lm_lstm_ptb', model_version='small')
    _model.train(epoch=150,
                 data_train=_data_train,
                 data_valid=_data_valid,
                 data_test=_data_test,
                 progress_interval=20)

