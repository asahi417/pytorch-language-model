""" language model training/evaluation"""
import argparse
import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import detect_anomaly
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from util import create_log, ParameterManager
from util_data import BatchFeeder, get_data, VALID_DATA_LIST, VALID_TOKENIZER_LIST
from util_hf_optimizer import AdamW, get_linear_schedule_with_warmup, get_constant_schedule


EPS = 1e-7


class LanguageModel:
    """ language model """

    def __init__(self,
                 model_type: str = 'lstm',
                 checkpoint: str = None,
                 checkpoint_dir: str = None,
                 default_parameter: str = None,
                 **kwargs):
        """ language model """
        self.__logger = create_log()
        self.__logger.debug('initialize network: *** %s based language model ***' % model_type)

        # setup parameter
        self.param = ParameterManager(
            checkpoint=checkpoint,
            checkpoint_dir=checkpoint_dir,
            default_parameter=default_parameter,
            **kwargs)
        self.checkpoint_model = os.path.join(self.param.checkpoint_dir, 'model.pt')

        # fix random seed
        random.seed(self.param('random_seed'))
        np.random.seed(self.param('random_seed'))
        torch.manual_seed(self.param('random_seed'))

        # build network
        if model_type == 'lstm':
            from model_lstm import StackedLSTM
            self.net = StackedLSTM(
                dropout_word=self.param("dropout_word"),
                dropout_embedding=self.param("dropout_embedding"),
                dropout_intermediate=self.param("dropout_intermediate"),
                dropout_output=self.param("dropout_output"),
                vocab_size=self.param("vocab_size"),
                embedding_dim=self.param("embedding_dim"),
                n_layers=self.param("n_layers"),
                n_hidden_units=self.param("n_hidden_units"),
                n_context=self.param("n_context"),
                tie_weights=self.param("tie_weights"),
                init_range=self.param("init_range")
            )
        elif model_type == 'gpt2':
            from model_gpt2 import GPT2
            self.net = GPT2(
                n_layer=self.param("n_layer"),
                n_embedding=self.param("n_embedding"),
                n_state_ffn=self.param("n_state_ffn"),
                n_head=self.param("n_head"),
                n_context=self.param("n_context"),
                dropout_residual=self.param("dropout_residual"),
                dropout_attention=self.param("dropout_attention"),
                dropout_embedding=self.param("dropout_embedding"),
                vocab_size=self.param("vocab_size")
            )
        elif model_type == 'transformer_xl':
            from model_transformer_xl import TransformerXL
            self.net = TransformerXL(
                n_layer=self.param("n_layer"),
                n_embedding=self.param("n_embedding"),
                n_state_ffn=self.param("n_state_ffn"),
                n_head=self.param("n_head"),
                n_context=self.param("n_context"),
                dropout_residual=self.param("dropout_residual"),
                dropout_attention=self.param("dropout_attention"),
                dropout_embedding=self.param("dropout_embedding"),
                vocab_size=self.param("vocab_size"),
                n_positional_embedding=self.param("n_positional_embedding"),
                initializer_range=self.param("init_range")
            )
        else:
            raise ValueError('bad model_type: %s' % model_type)

        # GPU allocation
        if torch.cuda.device_count() == 1:
            self.__logger.debug('running on single GPU')
            self.net = self.net.cuda()
            self.n_gpu = 1
            self.device = torch.device('cuda')
        elif torch.cuda.device_count() > 1:
            self.__logger.debug('running on %i GPUs' % torch.cuda.device_count())
            self.net = torch.nn.DataParallel(self.net.cuda())
            self.n_gpu = torch.cuda.device_count()
            self.device = torch.device('cuda')
        else:
            self.__logger.debug('no GPUs found')
            self.n_gpu = 0
            self.device = torch.device('cpu')

        # optimizer
        if self.param("optimizer") == 'adamw':
            self.optimizer = AdamW(
               self.net.parameters(), lr=self.param('lr'), weight_decay=self.param('weight_decay'))
        elif self.param("optimizer") == 'adam':
            self.optimizer = optim.Adam(
                self.net.parameters(), lr=self.param('lr'), weight_decay=self.param('weight_decay'))
        elif self.param("optimizer") == 'sgd':
            self.optimizer = optim.SGD(
                self.net.parameters(), lr=self.param('lr'), weight_decay=self.param('weight_decay'))
        else:
            raise ValueError('bad optimizer: %s' % self.param("optimizer"))
        if self.param('scheduler') == 'constant':
            self.scheduler = get_constant_schedule(self.optimizer)
        elif self.param('scheduler') == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.param('warmup_steps'),
                num_training_steps=self.param('total_steps'))
        else:
            raise ValueError('bad scheduler: %s' % self.param('scheduler'))

        # loss definition (CrossEntropyLoss includes softmax inside)
        # self.__loss = nn.CrossEntropyLoss()
        self.__loss = nn.NLLLoss()

        # load pre-trained ckpt
        if os.path.exists(self.checkpoint_model):
            ckpt = torch.load(self.checkpoint_model, map_location=self.device)
            self.net.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            self.__training_step = ckpt['training_step']  # num of training step
            self.__epoch = ckpt['epoch']
            self.__best_epoch = ckpt['best_epoch']
            self.__best_val_ppl = ckpt['best_val_ppl']
            self.__logger.debug('load ckpt from %s' % self.checkpoint_model)
            self.__logger.debug(' - epoch (best): %s (%s) ' % (str(ckpt['epoch']), str(ckpt['best_epoch'])))
            self.__logger.debug(' - ppl (best)  : %s (%s) ' % (str(ckpt['val_ppl']), str(ckpt['best_val_ppl'])))
        else:
            self.__training_step = 0
            self.__epoch = 0
            self.__best_epoch = 0
            self.__best_val_ppl = None

        # log
        self.__writer = SummaryWriter(log_dir=self.param.checkpoint_dir)
        self.__sanity_check()
        self.__model_type = model_type

    @property
    def hyperparameters(self):
        return self.param

    def __sanity_check(self):
        """ sanity check as logging model size """
        self.__logger.debug('trainable variables')
        model_size = 0
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                __shape = list(param.data.shape)
                model_size += np.prod(__shape)
                self.__logger.debug(' - [weight size] %s: %s' % (name, str(__shape)))
        self.__logger.debug(' - %i variables in total' % model_size)
        self.__logger.debug('hyperparameters')
        for k, v in self.param.parameter.items():
            self.__logger.debug(' - [param] %s: %s' % (k, str(v)))

    def evaluate(self, data_valid, data_test=None, n_extra_context: int = None):
        """ evaluate model """

        for d, n in zip([data_valid, data_test], ['valid', 'test']):
            if d is None:
                continue
            loss, ppl, bpc = self.__epoch_valid(
                BatchFeeder(sequence=d, batch_size=1, num_steps=self.param('n_context')),
                is_test=True,
                n_extra_context=n_extra_context
            )
            self.__logger.debug('(eval.%s) loss: %.5f, ppl: %.5f, bpc: %.5f' % (n, loss, ppl, bpc))

    def train(self,
              data_train: list,
              data_valid: list,
              data_test: list=None,
              progress_interval: int = 100):
        """ train model """
        best_model_wts = None
        val_ppl = None

        self.__logger.debug('initialize batch feeder')
        batch_param = dict(batch_size=self.param('batch_size'), num_steps=self.param('n_context'))
        if self.__model_type == 'transformer_xl':
            batch_param_valid = dict(batch_size=int(self.param('batch_size')/2), num_steps=self.param('n_context'))
        else:
            batch_param_valid = batch_param
        loader_train = BatchFeeder(sequence=data_train, **batch_param)
        loader_valid = BatchFeeder(sequence=data_valid, **batch_param_valid)

        try:
            with detect_anomaly():
                while True:
                    loss, ppl, bpc = self.__epoch_train(loader_train, progress_interval=progress_interval)
                    val_loss, val_ppl, val_bpc = self.__epoch_valid(loader_valid)
                    self.__logger.debug(
                        '[epoch %i] (train) loss: %.3f, ppl: %.3f, bpc: %.3f (valid) loss: %.3f, ppl: %.3f, bpc: %.3f'
                        % (self.__epoch, loss, ppl, bpc, val_loss, val_ppl, val_bpc))

                    if self.__best_val_ppl is None or val_ppl < self.__best_val_ppl:
                        best_model_wts = copy.deepcopy(self.net.state_dict())
                        self.__best_epoch = self.__epoch
                        self.__best_val_ppl = val_ppl

                    # TODO: Fix as this cant stop till the epoch done
                    if self.__training_step > self.param('total_steps'):
                        if data_test:
                            loader_test = BatchFeeder(sequence=data_test, **batch_param_valid)
                            loss, ppl, bpc = self.__epoch_valid(loader_test, is_test=True)
                            self.__logger.debug('(test) loss: %.3f, ppl: %.3f, bpc: %.3f' % (loss, ppl, bpc))
                        break

        except KeyboardInterrupt:
            self.__logger.info('*** KeyboardInterrupt ***')

        if self.__best_val_ppl is None:
            exit('nothing to be saved')

        self.__logger.debug('[training completed] best model: valid ppt %0.3f at epoch %i'
                            % (self.__best_val_ppl, self.__best_epoch))
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_step': self.__training_step,
            'epoch': self.__epoch,
            'val_ppl': val_ppl,
            'best_epoch': self.__best_epoch,
            'best_val_ppl': self.__best_val_ppl,
            'best_model_state_dict': best_model_wts,
        }, self.checkpoint_model)
        self.__writer.close()
        self.__logger.info('ckpt saved at %s' % self.checkpoint_model)

    def __epoch_train(self, data_loader, progress_interval: int = 100000):
        """ single epoch process for training """
        self.net.train()
        perplexity = None
        bpc = None
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
            self.optimizer.zero_grad()
            # forward: output prediction and get loss
            if self.__model_type == 'lstm':
                if self.param('pass_state'):
                    (_, prob, pred), hidden_state = self.net(inputs, hidden_state)
                else:
                    (_, prob, pred), _ = self.net(inputs)
            elif self.__model_type == 'transformer_xl':
                (_, prob, pred), hidden_state = self.net(inputs, hidden_state, self.param('n_context_memory'))
            else:
                _, prob, pred = self.net(inputs)

            # backward: calculate gradient
            prob = prob.add(EPS)
            prob = prob.view(-1, prob.size(-1))
            outputs = outputs.view(-1)
            tmp_loss = self.__loss(prob, outputs)
            tmp_loss.backward()
            # gradient clip
            if self.param('clip') is not None:
                nn.utils.clip_grad_norm_(self.net.parameters(), self.param('clip'))
            # optimize
            self.optimizer.step()
            self.scheduler.step()
            # log
            full_loss += len(outputs) * tmp_loss.cpu().item()
            full_seq_length += len(outputs)
            perplexity = np.exp(min(30, full_loss / full_seq_length))
            bpc = min(30, full_loss / full_seq_length) / np.log(2)
            mean_loss = full_loss / full_seq_length
            lr = self.optimizer.param_groups[0]['lr']
            self.__writer.add_scalar('train/loss', mean_loss, self.__training_step)
            self.__writer.add_scalar('train/perplexity', perplexity, self.__training_step)
            self.__writer.add_scalar('train/bpc', bpc, self.__training_step)
            self.__writer.add_scalar('learning_rate', lr, self.__training_step)

            if self.__training_step % progress_interval == 0:
                self.__logger.debug(' * (step %i) ppl: %.3f, bpc: %.3f, lr: %0.6f'
                                    % (self.__training_step, perplexity, bpc, lr))

            self.__training_step += 1
        self.__epoch += 1
        return mean_loss, perplexity, bpc

    def __epoch_valid(self, data_loader, is_test: bool=False, n_extra_context: int = None):
        """ validation/test """
        self.net.eval()
        full_seq_length = 0
        full_loss = 0
        hidden_state = None
        for data in data_loader:
            inputs, outputs = data
            if self.n_gpu > 0:
                inputs, outputs = inputs.cuda(), outputs.cuda()

            if self.__model_type == 'lstm':
                (logit, prob, pred), hidden_state = self.net(inputs, hidden_state)
            elif self.__model_type == 'transformer_xl':
                (logit, prob, pred), hidden_state = self.net(
                    inputs,
                    hidden_state,
                    n_extra_context if n_extra_context is not None else self.param('n_context_memory'))
                # print(hidden_state)
            else:
                logit, prob, pred = self.net(inputs)

            logit = logit.view(-1, logit.size(-1))
            outputs = outputs.view(-1)
            full_loss += len(outputs) * self.__loss(logit, outputs).cpu().item()
            full_seq_length += len(outputs)
        mean_loss = full_loss / full_seq_length
        perplexity = np.exp(min(30, full_loss / full_seq_length))
        bpc = min(30, full_loss / full_seq_length) / np.log(2)
        if not is_test:
            self.__writer.add_scalar('valid/perplexity', perplexity, self.__epoch)
            self.__writer.add_scalar('valid/bpc', bpc, self.__epoch)
            self.__writer.add_scalar('valid/loss', mean_loss, self.__epoch)
        return mean_loss, perplexity, bpc


def get_options():
    parser = argparse.ArgumentParser(description='Train language model', formatter_class=argparse.RawTextHelpFormatter)
    _p = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-c', '--ckpt', help='pre-trained model ckpt', default=None, type=str, **_p)
    parser.add_argument('-e', '--evaluate', help='evaluation', action='store_true')
    parser.add_argument('-d', '--data', help='data: %s' % str(VALID_DATA_LIST), default='PennTreebank', type=str, **_p)
    parser.add_argument('-t', '--tokenizer', help='tokenizer: %s' % str(VALID_TOKENIZER_LIST), default='SentencePieceBPETokenizer', type=str, **_p)
    parser.add_argument('-m', '--model', help='model', default='lstm', type=str, **_p)
    parser.add_argument('--n_extra_context', help='n_extra_context for transformer XL (eval mode)', default=None, type=int, **_p)
    return parser.parse_args()


if __name__ == '__main__':
    arguments = get_options()
    # vocab size can be applied for tokenizer except Whitespace
    _data_train, _data_valid, _data_test = get_data(arguments.data, arguments.tokenizer, vocab_size=10000)

    config_path = os.path.join(arguments.data, arguments.tokenizer, arguments.model)

    model_instance = LanguageModel(
        model_type=arguments.model,
        checkpoint=arguments.ckpt,
        checkpoint_dir=os.path.join('./ckpt', config_path),
        default_parameter=os.path.join('./parameters', config_path) + '.toml')

    if arguments.evaluate:
        model_instance.evaluate(data_valid=_data_valid,
                                data_test=_data_test,
                                n_extra_context=arguments.n_extra_context)

    else:
        model_instance.train(data_train=_data_train,
                             data_valid=_data_valid,
                             data_test=_data_test,
                             progress_interval=20)

