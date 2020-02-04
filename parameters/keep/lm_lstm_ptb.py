""" pytorch LSTM based language model """

# for model
import os
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from util import create_log, ParameterManager
from util_data import BatchFeeder, get_data


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
        self.__decoding_layer = nn.Linear(embedding_dim, vocab_size, bias=False)

        if tie_weights:
            # nn.Embedding(a, b).weight.shape -> (a, b), while nn.Linear(a, b) -> (b, a)
            # so encoder's weight can be directly copied to decoder.
            self.__decoding_layer.weight = self.__embedding_layer.weight

        self.__sequence_length = sequence_length
        self.__n_layers = n_layers
        self.__n_hidden_units = n_hidden_units
        self.__embedding_dim = embedding_dim
        self.__tie_weights = tie_weights
        self.__vocab_size = vocab_size
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
            if torch.cuda.device_count() >= 1:
                state = [torch.zeros((1, batch_size, units), dtype=torch.float32).cuda(),
                         torch.zeros((1, batch_size, units), dtype=torch.float32).cuda()]
            else:
                state = [torch.zeros((1, batch_size, units), dtype=torch.float32),
                         torch.zeros((1, batch_size, units), dtype=torch.float32)]
            return state

        return [__init_state(i) for i in range(self.__n_layers)]

    def forward(self, input_token, hidden=None):
        """ model output

         Parameter
        -------------
        input_token: input token id batch tensor (batch, sequence_length)
        hidden: list of two tensors, each has (layer, batch, dim) shape

         Return
        -------------
        (output, prob, pred):
            output: raw output from LSTM (sequence_length, batch, vocab size)
            prob: softmax activated output (sequence_length, batch, vocab size)
            pred: prediction (sequence_length, batch)
        new_hidden: list of tensor (layer, batch, dim)
        """
        # (batch, sequence_length) -> (sequence_length, batch)
        input_token = input_token.permute(1, 0).contiguous()
        if hidden is None:
            hidden = self.init_state(input_token.shape[1])
        # print([i.shape for i in hidden])
        emb = self.__embedding_lookup(self.__embedding_layer, input_token)  # lookup embedding matrix (seq, batch, dim)
        emb = self.__dropout_embedding(emb)  # dropout embeddings
        new_hidden = []  # hidden states

        for i, (h, cell) in enumerate(zip(hidden, self.__cells)):
            # LSTM input is (sequence, batch, dim)
            emb, new_h = cell(emb, h)
            # detach hidden state from the graph to not propagate gradient (treat as a constant)
            new_h = self.repackage_hidden(new_h)
            new_hidden.append(new_h)
            if i == self.__n_layers - 1:
                emb = self.__dropout_output(emb)
            else:
                emb = self.__dropout_intermediate(emb)

        # (seq, batch, dim) -> (seq * batch, dim)
        output = emb.view(emb.size(0) * emb.size(1), emb.size(2))
        # (seq * batch, dim) -> (seq * batch, vocab)
        output = self.__decoding_layer(output)
        _, pred = torch.max(output, dim=-1)
        prob = torch.nn.functional.softmax(output, dim=1)
        # (seq * batch, vocab) -> (seq, batch, vocab)
        output = output.view(emb.size(0), emb.size(1), self.__vocab_size)
        prob = prob.view(emb.size(0), emb.size(1), self.__vocab_size)
        # (seq * batch, vocab) -> (seq, batch,)
        pred = pred.view(emb.size(0), emb.size(1))
        return (output, prob, pred), new_hidden

    def repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""

        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)


class LanguageModel:
    """ LSTM bases language model """

    def __init__(self,
                 checkpoint: str = None,
                 checkpoint_dir: str = None,
                 default_parameter: str = None,
                 **kwargs):
        """ LSTM bases language model """
        self.__logger = create_log()
        self.__logger.debug('initialize network: *** LSTM based language model ***')
        # setup parameter
        self.__param = ParameterManager(
            checkpoint=checkpoint,
            checkpoint_dir=checkpoint_dir,
            default_parameter=default_parameter,  # './parameters/lm_lstm_ptb.toml',
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
        if torch.cuda.device_count() == 1:
            self.__logger.debug('running on single GPU')
            self.__net = self.__net.cuda()
            self.n_gpu = 1
        elif torch.cuda.device_count() > 1:
            self.__logger.debug('running on %i GPUs' % torch.cuda.device_count())
            self.__net = torch.nn.DataParallel(self.__net.cuda())
            self.n_gpu = torch.cuda.device_count()
        else:
            self.__logger.debug('no GPUs found')
            self.n_gpu = 0

        # optimizer
        self.__optimizer = optim.Adam(
            self.__net.parameters(), lr=self.__param('lr'), weight_decay=self.__param('weight_decay'))

        # loss definition (CrossEntropyLoss includes softmax inside)
        self.__loss = nn.CrossEntropyLoss()

        # load pre-trained ckpt
        if os.path.exists(self.__checkpoint_model):
            ckpt = torch.load(self.__checkpoint_model)
            self.__net.load_state_dict(ckpt['model_state_dict'])
            self.__optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.__training_step = ckpt['training_step']  # num of training step
            self.__epoch = ckpt['epoch']
            self.__best_epoch = ckpt['best_epoch']
            self.__best_val_ppl = ckpt['best_val_ppl']
            self.__logger.debug('load ckpt from %s' % self.__checkpoint_model)
            self.__logger.debug(' - epoch (best): %s (%s) ' % (str(ckpt['epoch']), str(ckpt['best_epoch'])))
            self.__logger.debug(' - ppl (best)  : %s (%s) ' % (str(ckpt['val_ppl']), str(ckpt['best_val_ppl'])))
        else:
            self.__training_step = 0
            self.__epoch = 0
            self.__best_epoch = 0
            self.__best_val_ppl = None

        # log
        self.__writer = SummaryWriter(log_dir=self.__param.checkpoint_dir)
        self.__sanity_check(self.__param('random_seed'))

    @property
    def hyperparameters(self):
        return self.__param

    def __sanity_check(self, seed: int = 1234):
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

        # fix random seed
        if seed:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if self.n_gpu > 0:
                torch.cuda.manual_seed_all(seed)

    # def evaluate(self, data_valid, data_test=None):
    #     """ evaluate model """
    #     batch_param = dict(batch_size=self.__param('batch_size'), num_steps=self.__param('n_context'))
    #     loss, ppl = self.__epoch_valid(BatchFeeder(sequence=data_valid, **batch_param))
    #     self.__logger.debug('(val)  loss: %.5f, ppl: %.5f' % (loss, ppl))
    #     if data_test:
    #         loss, ppl = self.__epoch_valid(BatchFeeder(sequence=data_test, **batch_param), is_test=True)
    #         self.__logger.debug('(test) loss: %.5f, ppl: %.5f' % (loss, ppl))

    def train(self,
              data_train: list,
              data_valid: list,
              data_test: list=None,
              progress_interval: int = 100):
        """ train model """
        best_model_wts = None
        val_ppl = None

        self.__logger.debug('initialize batch feeder')
        batch_param = dict(batch_size=self.__param('batch_size'), num_steps=self.__param('n_context'))
        loader_train = BatchFeeder(sequence=data_train, **batch_param)
        loader_valid = BatchFeeder(sequence=data_valid, **batch_param)

        try:
            while True:
                loss, ppl = self.__epoch_train(loader_train, progress_interval=progress_interval)
                val_loss, val_ppl = self.__epoch_valid(loader_valid)
                self.__logger.debug('[epoch %i] (train) loss: %.3f, ppl: %.3f (valid) loss: %.3f, ppl: %.3f'
                                    % (self.__epoch, loss, ppl, val_loss, val_ppl))

                if self.__best_val_ppl is None or val_ppl < self.__best_val_ppl:
                    best_model_wts = copy.deepcopy(self.__net.state_dict())
                    self.__best_epoch = self.__epoch
                    self.__best_val_ppl = val_ppl

                if self.__training_step > self.__param('total_steps'):
                    if data_test:
                        loader_test = BatchFeeder(sequence=data_test, **batch_param)
                        loss, ppl = self.__epoch_valid(loader_test, is_test=True)
                        self.__logger.debug('(test) loss: %.3f, ppl: %.3f' % (loss, ppl))
                    break

        except KeyboardInterrupt:
            self.__logger.info('*** KeyboardInterrupt ***')

        self.__logger.debug('[training completed] best model: valid ppt %0.3f at epoch %i'
                            % (self.__best_val_ppl, self.__best_epoch))
        torch.save({
            'model_state_dict': self.__net.state_dict(),
            'optimizer_state_dict': self.__optimizer.state_dict(),
            'training_step': self.__training_step,
            'epoch': self.__epoch,
            'val_ppl': val_ppl,
            'best_epoch': self.__best_epoch,
            'best_val_ppl': self.__best_val_ppl,
            'best_model_state_dict': best_model_wts,
        }, self.__checkpoint_model)
        self.__writer.close()
        self.__logger.info('ckpt saved at %s' % self.__checkpoint_model)

    def __epoch_train(self, data_loader, progress_interval: int = 100000):
        """ single epoch process for training """
        self.__net.train()
        perplexity = None
        mean_loss = None
        full_seq_length = 0
        full_loss = 0
        hidden_state = None

        for i, data in enumerate(data_loader, 1):

            # get the inputs (data is a list of [inputs, labels])
            inputs, outputs = data
            if self.n_gpu > 0:
                inputs, outputs = inputs.cuda(), outputs.cuda()
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
            full_loss += len(outputs) * tmp_loss.cpu().item()
            full_seq_length += len(outputs)
            perplexity = np.exp(min(30, full_loss / full_seq_length))
            mean_loss = full_loss / full_seq_length
            lr = self.__optimizer.param_groups[0]['lr']
            self.__writer.add_scalar('train/loss', mean_loss, self.__training_step)
            self.__writer.add_scalar('train/perplexity', perplexity, self.__training_step)
            self.__writer.add_scalar('learning_rate', lr, self.__training_step)

            if self.__training_step % progress_interval == 0:
                self.__logger.debug(' * (step %i) ppl: %.3f, lr: %0.6f' % (self.__training_step, perplexity, lr))

            self.__training_step += 1
        self.__epoch += 1
        return mean_loss, perplexity

    def __epoch_valid(self, data_loader, is_test: bool=False):
        """ validation/test """
        self.__net.eval()
        full_seq_length = 0
        full_loss = 0
        hidden_state = None
        for data in data_loader:
            inputs, outputs = data
            (logit, prob, pred), hidden_state = self.__net(inputs, hidden_state)
            logit = logit.view(-1, logit.size(-1))
            outputs = outputs.view(-1)
            full_loss += len(outputs) * self.__loss(logit, outputs).cpu().item()
            full_seq_length += len(outputs)
        mean_loss = full_loss / full_seq_length
        perplexity = np.exp(min(30, full_loss / full_seq_length))
        if not is_test:
            self.__writer.add_scalar('valid/perplexity', perplexity, self.__epoch)
            self.__writer.add_scalar('valid/loss', mean_loss, self.__epoch)
        return mean_loss, perplexity


def get_options():
    parser = argparse.ArgumentParser(description='Train tokenizer', formatter_class=argparse.RawTextHelpFormatter)
    _p = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-c', '--ckpt', help='pre-trained model ckpt', default=None, type=str, **_p)
    parser.add_argument('-e', '--evaluate', help='evaluation', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    with open('./data/penn-treebank/ptb.train.eos.id.txt', 'r') as f:
        _data_train = [int(i) for i in f.read().split()]
    with open('./data/penn-treebank/ptb.valid.eos.id.txt', 'r') as f:
        _data_valid = [int(i) for i in f.read().split()]
    with open('./data/penn-treebank/ptb.test.eos.id.txt', 'r') as f:
        _data_test = [int(i) for i in f.read().split()]

    _model = LanguageModel(checkpoint=arguments.ckpt,
                           checkpoint_dir='./ckpt/lm_lstm_ptb',
                           default_parameter='./parameters/lm_lstm_ptb.toml')
    _model.train(epoch=150,
                 data_train=_data_train,
                 data_valid=_data_valid,
                 data_test=_data_test,
                 progress_interval=20)

