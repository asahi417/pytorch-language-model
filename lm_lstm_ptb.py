""" pytorch sample of language model

* locked (variational) dropout
* logging instance loss/accuracy with progress interval
* checkpoint manager
* save the best model in terms of valid accuracy
* tensorboard
"""

# for model
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# for parameter manager
import json
import os
import random
import string
import toml
from glob import glob

# for logger
import logging
from logging.config import dictConfig


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
                        os.remove(_dir)
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


class LockedDropout(nn.Module):
    """ locked dropout/variational dropout described in https://arxiv.org/pdf/1708.02182.pdf
    * drop all the feature in target batch, instead of fully random dropout
    """

    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.__dropout = dropout

    def forward(self, x):
        if not self.training or not self.__dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2))
        m = m.bernoulli_(1 - self.__dropout)  # rescale un-dropped values to keep distribution consistent
        mask = Variable(m, requires_grad=False) / (1 - self.__dropout)
        mask = mask.expand_as(x)
        return mask * x


class EmbeddingLookup(nn.Module):
    """ Embedding lookup layer with word dropout described in https://arxiv.org/pdf/1708.02182.pdf
    * drop all the embedding in target word, instead of fully random dropout
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.__dropout = dropout

    def forward(self, embedding_mat, words):
        if not self.training or not self.__dropout:
            masked_embed_weight = embedding_mat.weight
        else:
            mask = embedding_mat.weight.data.new().resize_((embedding_mat.weight.size(0), 1))
            # rescale un-dropped values to keep distribution consistent
            mask = mask.bernoulli_(1 - self.__dropout).expand_as(embedding_mat.weight) / (1 - self.__dropout)
            masked_embed_weight = mask * embedding_mat.weight

        # lookup embedding with mask
        x = nn.functional.embedding(words,
                                    weight=masked_embed_weight,
                                    padding_idx=-1 if embedding_mat.padding_idx is None else embedding_mat.padding_idx,
                                    max_norm=embedding_mat.max_norm,
                                    norm_type=embedding_mat.norm_type,
                                    scale_grad_by_freq=embedding_mat.scale_grad_by_freq,
                                    sparse=embedding_mat.sparse)
        return x


class Net(nn.Module):
    """ Network Architecture: LSTM based Language Model """

    def __init__(self,
                 dropout_word: float,
                 dropout_embedding: float,
                 dropout_intermediate: float,
                 dropout_output: float,
                 vocab_size: int,
                 embedding_dim: int,
                 n_layers: int,
                 n_hidden_units: int,
                 sequence_length: int,
                 tie_weights: bool,
                 init_range: float):
        """ Network Architecture """
        super(Net, self).__init__()
        self.__embedding_lookup = EmbeddingLookup(dropout_word)
        self.__dropout_embedding = LockedDropout(dropout_embedding)
        self.__dropout_intermediate = LockedDropout(dropout_intermediate)
        self.__dropout_output = LockedDropout(dropout_output)

        cells = []
        for i in range(n_layers):
            if i == 0:
                cell = nn.LSTM(embedding_dim, n_hidden_units)
            elif i == n_layers - 1:
                cell = nn.LSTM(n_hidden_units, embedding_dim)
            else:
                cell = nn.LSTM(n_hidden_units, n_hidden_units)
            cells.append(cell)

        self.__cells = nn.ModuleList(cells)

        self.__embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.__decoding_layer = nn.Linear(embedding_dim, vocab_size)

        if tie_weights:
            # nn.Embedding(a, b).weight.shape -> (a, b), while nn.Linear(a, b) -> (b, a)
            # so encoder's weight can be directly copied to decoder.
            self.__decoding_layer.weight = self.__embedding_layer.weight

        self.__sequence_length = sequence_length
        self.__n_layers = n_layers
        self.__n_hidden_units = n_hidden_units
        self.__embedding_dim = embedding_dim
        self.__tie_weights = tie_weights
        self.__init_weights(init_range=init_range)

    def __init_weights(self, init_range: float):
        """ uniform weight initialization for encoding/decoding layer """
        self.__embedding_layer.weight.data.uniform_(-init_range, init_range)
        if not self.__tie_weights:
            self.__decoding_layer.weight.data.uniform_(-init_range, init_range)

    def init_state(self, batch_size: int):
        """ get initial state of recurrent cell: list of tensor (layer, batch, dim) """

        def __init_state(i):
            if i == self.__n_layers - 1:
                units = self.__embedding_dim
            else:
                units = self.__n_hidden_units
            state = torch.zeros((1, batch_size, units), dtype=torch.float32)
            if torch.cuda.device_count() >= 1:
                state.cuda()
            return state

        return [__init_state(i) for i in range(self.__n_layers)]

    def forward(self, input_token, hidden=None):
        """ model output

         Parameter
        -------------
        input_token: input token id batche tensor (batch, )
        hidden: list of tensor (layer, batch, dim)

         Return
        -------------
        x: logit (batch, dim)
        y: prediction (batch, )
        p: probability (batch, dim)
        """

        if hidden is None:
            hidden = self.init_state(input_token.shape[0])

        emb = self.__embedding_lookup(self.__embedding_layer, input_token)  # lookup embedding matrix
        emb = self.__dropout_embedding(emb)  # dropout embeddings
        new_hidden = []  # hidden states

        for i, cell in enumerate(self.__cells):
            emb, new_h = cell(emb, hidden[i])
            new_hidden.append(new_h)
            if i == self.__n_layers - 1:
                emb = self.__dropout_output(emb)
            else:
                emb = self.__dropout_intermediate(emb)

        output = emb.view(emb.size(0) * emb.size(1), emb.size(2))
        return output, new_hidden


class LanguageModel:
    """ LSTM bases language model """

    def __init__(self,
                 progress_interval: int = 20000,
                 checkpoint_dir: str = None,
                 **kwargs):
        """ LSTM bases language model
        * Allocate a GPU automatically; specify by CUDA_VISIBLE_DEVICES
        * Load checkpoints if it exists
        """
        self.__logger = create_log()
        self.__logger.debug('initialize network: \n*** LSTM based language model ***\n')
        # setup parameter
        self.__param = ParameterManager(
            checkpoint_dir=checkpoint_dir,
            default_parameter='./parameters/lm_lstm_ptb.toml',
            **kwargs)
        self.__checkpoint_model = os.path.join(self.__param.checkpoint_dir, 'model.pt')
        # build network
        self.__net = Net(
            dropout_word=self.__param("dropout_word"),
            dropout_embedding=self.__param("dropout_embedding"),
            dropout_intermediate=self.__param("dropout_intermediate"),
            dropout_output=self.__param("dropout_output"),
            vocab_size=self.__param("vocab_size"),
            embedding_dim=self.__param("embedding_dim"),
            n_layers=self.__param("n_layers"),
            n_hidden_units=self.__param("n_hidden_units"),
            sequence_length=self.__param("sequence_length"),
            tie_weights=self.__param("tie_weights"),
            init_range=self.__param("init_range")
        )
        # GPU allocation
        if torch.cuda.device_count() >= 1:
            self.__logger.debug('running on GPU')
            self.if_use_gpu = True
            self.__net = self.__net.cuda()
        else:
            self.if_use_gpu = False
        # optimizer
        self.__optimizer = optim.Adam(
            self.__net.parameters(), lr=self.__param('lr'), weight_decay=self.__param('weight_decay'))

        # loss definition (CrossEntropyLoss includes softmax inside)
        self.__loss = nn.CrossEntropyLoss()

        # load pre-trained ckpt
        if os.path.exists(self.__checkpoint_model):
            self.__net.load_state_dict(torch.load(self.__checkpoint_model))
            self.__logger.debug('load ckpt from %s' % self.__checkpoint_model)
        # log
        self.__progress_interval = progress_interval
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
              data_train,
              data_valid,
              epoch: int):
        """ train model """
        best_model_wts = copy.deepcopy(self.__net.state_dict())
        best_acc = 0
        best_epoch = 0

        assert data_train.classes == data_valid.classes
        assert len(data_train.classes) == self.__param('label_size')

        self.__logger.debug('data loader instance')
        loader_train = torch.utils.data.DataLoader(
            data_train, batch_size=self.__param('batch_size'), shuffle=True, num_workers=4)
        loader_valid = torch.utils.data.DataLoader(
            data_valid, batch_size=self.__param('batch_size'), shuffle=False, num_workers=4)

        try:
            for e in range(epoch):  # loop over the epoch

                loss, acc = self.__epoch_train(loader_train, epoch_n=e)
                self.__logger.debug('[epoch %i/%i] (train) loss: %.3f, acc: %.3f' % (e, epoch, loss, acc))

                loss, acc = self.__epoch_valid(loader_valid, epoch_n=e)
                self.__logger.debug('[epoch %i/%i] (valid) loss: %.3f, acc: %.3f' % (e, epoch, loss, acc))

                if acc > best_acc:
                    best_model_wts = copy.deepcopy(self.__net.state_dict())
                    best_epoch = e
                    best_acc = acc

        except KeyboardInterrupt:
            self.__logger.info('*** KeyboardInterrupt ***')

        self.__writer.close()
        self.__logger.debug('best model: epoch %i, valid accuracy %0.3f' % (best_epoch, best_acc))
        torch.save(best_model_wts, self.__checkpoint_model)
        self.__logger.debug('complete training: best model ckpt was saved at %s' % self.__checkpoint_model)

    def __epoch_train(self,
                      data_loader,
                      epoch_n: int):
        """ single epoch process for training """
        self.__net.train()
        mean_loss = 0.0
        correct_count = 0.0
        data_size = 0.0
        inst_loss = 0.0

        for i, data in enumerate(data_loader, 1):
            # get the inputs (data is a list of [inputs, labels])
            inputs, labels = data
            if self.if_use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            # zero the parameter gradients
            self.__optimizer.zero_grad()
            # forward: output prediction and get loss
            logit = self.__net(inputs)
            _, pred = torch.max(logit, 1)

            tmp_loss = self.__loss(logit, labels)
            # backward: calculate gradient
            tmp_loss.backward()
            # gradient clip
            if self.__param('clip') is not None:
                nn.utils.clip_grad_norm_(self.__net.parameters(), self.__param('clip'))
            # optimize
            self.__optimizer.step()
            # accuracy
            correct_count += ((pred == labels).cpu().float().sum()).item()
            data_size += len(labels)

            # log
            loss = tmp_loss.cpu().item()
            self.__writer.add_scalar('train/loss', loss, i + epoch_n * len(data_loader))
            self.__writer.add_scalar('train/accuracy', correct_count / data_size, i + epoch_n * len(data_loader))
            inst_loss += loss
            mean_loss += loss

            if i % self.__progress_interval == 0:
                inst_loss = inst_loss / self.__progress_interval
                self.__logger.debug(' * (%i/%i) instant loss: %.3f' % (i, len(data_loader), inst_loss))
                inst_loss = 0.0

        mean_loss = mean_loss / len(data_loader)
        mean_accuracy = correct_count / data_size
        return mean_loss, mean_accuracy

    def __epoch_valid(self,
                      data_loader,
                      epoch_n: int):
        """ single epoch process for validation """
        self.__net.eval()
        mean_loss = 0.0
        correct_count = 0.0
        data_size = 0.0

        for data in data_loader:
            inputs, labels = data
            if self.if_use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            logit = self.__net(inputs)
            _, pred = torch.max(logit, 1)
            correct_count += ((pred == labels).cpu().float().sum()).item()
            mean_loss += self.__loss(logit, labels).cpu().item()
            data_size += len(labels)
        self.__writer.add_scalar('valid/loss', mean_loss / len(data_loader), epoch_n)
        self.__writer.add_scalar('valid/accuracy', correct_count / data_size, epoch_n)
        mean_loss = mean_loss / len(data_loader)
        mean_accuracy = correct_count / data_size
        return mean_loss, mean_accuracy


if __name__ == '__main__':
    data_train = './data/penn-treebank/ptb_train_id.txt'
    data_test = './data/penn-treebank/ptb_test_id.txt'
    data_valid = './data/penn-treebank/ptb_valid_id.txt'

    # LanguageModel(checkpoint_dir='./ckpt/lm_lstm_ptb')
