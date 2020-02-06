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
        self.__param = ParameterManager(
            checkpoint=checkpoint,
            checkpoint_dir=checkpoint_dir,
            default_parameter=default_parameter,
            **kwargs)
        self.__checkpoint_model = os.path.join(self.__param.checkpoint_dir, 'model.pt')

        # build network
        if model_type == 'lstm':
            from module_lstm import StackedLSTM
            self.__net = StackedLSTM(
                dropout_word=self.__param("dropout_word"),
                dropout_embedding=self.__param("dropout_embedding"),
                dropout_intermediate=self.__param("dropout_intermediate"),
                dropout_output=self.__param("dropout_output"),
                vocab_size=self.__param("vocab_size"),
                embedding_dim=self.__param("embedding_dim"),
                n_layers=self.__param("n_layers"),
                n_hidden_units=self.__param("n_hidden_units"),
                n_context=self.__param("n_context"),
                tie_weights=self.__param("tie_weights"),
                init_range=self.__param("init_range")
            )
        elif model_type == 'gpt2':
            from module_transformer import GPT2
            self.__net = GPT2(
                n_layer=self.__param("n_layer"),
                n_embedding=self.__param("n_embedding"),
                n_state_ffn=self.__param("n_state_ffn"),
                n_head=self.__param("n_head"),
                n_context=self.__param("n_context"),
                max_cache_size=self.__param("max_cache_size"),
                residual_dropout=self.__param("residual_dropout"),
                attention_dropout=self.__param("attention_dropout"),
                embedding_dropout=self.__param("embedding_dropout"),
                vocab_size=self.__param("vocab_size")
            )
        else:
            raise ValueError('bad model_type: %s' % model_type)

        # GPU allocation
        if torch.cuda.device_count() == 1:
            self.__logger.debug('running on single GPU')
            self.__net = self.__net.cuda()
            self.n_gpu = 1
            self.device = torch.device('cuda')
        elif torch.cuda.device_count() > 1:
            self.__logger.debug('running on %i GPUs' % torch.cuda.device_count())
            self.__net = torch.nn.DataParallel(self.__net.cuda())
            self.n_gpu = torch.cuda.device_count()
            self.device = torch.device('cuda')
        else:
            self.__logger.debug('no GPUs found')
            self.n_gpu = 0
            self.device = torch.device('cpu')

        # optimizer
        if self.__param("optimizer") == 'adamw':
            self.__optimizer = AdamW(
               self.__net.parameters(), lr=self.__param('lr'), weight_decay=self.__param('weight_decay'))
        elif self.__param("optimizer") == 'sgd':
            self.__optimizer = optim.SGD(
                self.__net.parameters(), lr=self.__param('lr'), weight_decay=self.__param('weight_decay'))
        else:
            raise ValueError('bad optimizer: %s' % self.__param("optimizer"))
        if self.__param('scheduler') == 'constant':
            self.__scheduler = get_constant_schedule(self.__optimizer)
        elif self.__param('scheduler') == 'linear':
            self.__scheduler = get_linear_schedule_with_warmup(
                self.__optimizer,
                num_warmup_steps=self.__param('warmup_steps'),
                num_training_steps=self.__param('total_steps'))
        else:
            raise ValueError('bad scheduler: %s' % self.__param('scheduler'))

        # loss definition (CrossEntropyLoss includes softmax inside)
        self.__loss = nn.CrossEntropyLoss()

        # load pre-trained ckpt
        if os.path.exists(self.__checkpoint_model):
            ckpt = torch.load(self.__checkpoint_model, map_location=self.device)
            self.__net.load_state_dict(ckpt['model_state_dict'])
            self.__optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.__scheduler.load_state_dict(ckpt['scheduler_state_dict'])
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

    def __sanity_check(self, seed: int):
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

    def evaluate(self, data_valid, data_test=None):
        """ evaluate model """
        with detect_anomaly():
            batch_param = dict(batch_size=self.__param('batch_size'), num_steps=self.__param('n_context'))
            loss, ppl, bpc = self.__epoch_valid(BatchFeeder(sequence=data_valid, **batch_param))
            self.__logger.debug('(val)  loss: %.5f, ppl: %.5f, bpc: %.5f' % (loss, ppl, bpc))
            if data_test:
                loss, ppl, bpc = self.__epoch_valid(BatchFeeder(sequence=data_test, **batch_param), is_test=True)
                self.__logger.debug('(test) loss: %.5f, ppl: %.5f, bpc: %.5f' % (loss, ppl, bpc))

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
            with detect_anomaly():
                while True:
                    loss, ppl, bpc = self.__epoch_train(loader_train, progress_interval=progress_interval)
                    val_loss, val_ppl, val_bpc = self.__epoch_valid(loader_valid)
                    self.__logger.debug(
                        '[epoch %i] (train) loss: %.3f, ppl: %.3f, bpc: %.3f (valid) loss: %.3f, ppl: %.3f, bpc: %.3f'
                        % (self.__epoch, loss, ppl, bpc, val_loss, val_ppl, val_bpc))

                    if self.__best_val_ppl is None or val_ppl < self.__best_val_ppl:
                        best_model_wts = copy.deepcopy(self.__net.state_dict())
                        self.__best_epoch = self.__epoch
                        self.__best_val_ppl = val_ppl

                    if self.__training_step > self.__param('total_steps'):
                        if data_test:
                            loader_test = BatchFeeder(sequence=data_test, **batch_param)
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
            'model_state_dict': self.__net.state_dict(),
            'optimizer_state_dict': self.__optimizer.state_dict(),
            'scheduler_state_dict': self.__scheduler.state_dict(),
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
            self.__scheduler.step()
            # log
            full_loss += len(outputs) * tmp_loss.cpu().item()
            full_seq_length += len(outputs)
            perplexity = np.exp(min(30, full_loss / full_seq_length))
            bpc = min(30, full_loss / full_seq_length) / np.log(2)
            mean_loss = full_loss / full_seq_length
            lr = self.__optimizer.param_groups[0]['lr']
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

    def __epoch_valid(self, data_loader, is_test: bool=False):
        """ validation/test """
        self.__net.eval()
        full_seq_length = 0
        full_loss = 0
        hidden_state = None
        for data in data_loader:
            inputs, outputs = data
            if self.n_gpu > 0:
                inputs, outputs = inputs.cuda(), outputs.cuda()
            (logit, prob, pred), hidden_state = self.__net(inputs, hidden_state)
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
    return parser.parse_args()


if __name__ == '__main__':
    arguments = get_options()
    # vocab size can be applied for tokenizer except Whitespace
    _data_train, _data_valid, _data_test = get_data(arguments.data, arguments.tokenizer, vocab_size=10000)

    config_path = os.path.join(arguments.data, arguments.tokenizer, arguments.model)

    model_instance = LanguageModel(
        checkpoint=arguments.ckpt,
        checkpoint_dir=os.path.join('./ckpt', config_path),
        default_parameter=os.path.join('./parameters', config_path) + '.toml')

    if arguments.evaluate:
        model_instance.evaluate(data_valid=_data_valid, data_test=_data_test)

    else:
        model_instance.train(data_train=_data_train,
                             data_valid=_data_valid,
                             data_test=_data_test,
                             progress_interval=20)

