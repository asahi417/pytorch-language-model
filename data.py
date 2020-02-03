""" Data process modules """
import os
import torch
import torchtext
import tokenizers
import logging
from logging.config import dictConfig


__all__ = [
    "get_tokenizer",
    "BatchFeeder"
]


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
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    return logger


def get_tokenizer(name: str, checkpoint_dir: str = None, checkpoint_name: str = None):
    """ get tokenizer instance"""
    if_trained = False
    if checkpoint_dir is not None and checkpoint_name is not None:
        merges = '%s-merges.txt' % os.path.join(checkpoint_dir, checkpoint_name)
        vocab = '%s-vocab.json' % os.path.join(checkpoint_dir, checkpoint_name)
        if os.path.exists(merges) and os.path.exists(vocab):
            if_trained = True
        else:
            merges = None
            vocab = None
    else:
        merges = None
        vocab = None

    if name == 'BPETokenizer':
        return tokenizers.BPETokenizer(vocab, merges), if_trained
    elif name == 'ByteLevelBPETokenizer':
        return tokenizers.ByteLevelBPETokenizer(vocab, merges), if_trained
    elif name == 'SentencePieceBPETokenizer':
        return tokenizers.SentencePieceBPETokenizer(vocab, merges), if_trained
    elif name == 'BertWordPieceTokenizer':
        return tokenizers.BertWordPieceTokenizer(vocab), if_trained
    else:
        raise ValueError('unknown tokenizer %s' % name)


def get_data(name,
             tokenizer: str = 'split',
             data_directory: str = './data',
             vocab_directory: str = './vocab',
             debug: bool = False):
    """ Get file path to tokenized benchmark data

     Parameter
    ------------
    name: dataset name
    tokenizer: tokenizer name
    data_directory: path to cache data
    vocab_directory: path to save tokenizer model

     Return
    ------------
    list of files (train/valid/test) for token_ids
    """
    logger = create_log()

    # fetch benchmark data
    def convert_eos(__file_path):
        __file_output = ''.join(__file_path.split('.')[:-1]) + '.eos.txt'
        if not os.path.exists(__file_output):
            with open(__file_path, 'r') as _f_r:
                with open(__file_output, 'w') as _f_w:
                    _f_w.write(_f_r.read().replace('\n', '<eos>'))
        return __file_output

    data_field = torchtext.data.Field(sequential=True)
    if name == 'PennTreebank':
        torchtext.datasets.PennTreebank.splits(data_field, root=data_directory)
        data_path = os.path.join(data_directory, 'penn-treebank')
        output_files = [
            convert_eos(os.path.join(data_path, _f)) for _f in ['ptb.train.txt', 'ptb.valid.txt', 'ptb.test.txt']]
    elif name == 'WikiText103':
        torchtext.datasets.WikiText103.splits(data_field, root=data_directory)
        data_path = os.path.join(data_directory, 'wikitext-103/wikitext-103')
        output_files = [
            convert_eos(os.path.join(data_path, _f)) for _f in ['ptb.train.txt', 'ptb.valid.txt', 'ptb.test.txt']]
    # elif name == 'enwiki8':
    #     pass
    else:
        raise ValueError('unknown data %s' % name)
    logger.debug('data %s has been downloaded' % name)

    # choose tokenizer
    if tokenizer == 'split':
        # WIP: better to implement basic tokenizer instance
        # logger.debug('set plain tokenizer (split by halfspace)')
        # get_tokenizer()
    else:
        save_dir = os.path.join(vocab_directory, tokenizer)
        tokenizer, if_trained_flg = get_tokenizer(tokenizer, checkpoint_dir=save_dir, checkpoint_name=name)
        if not if_trained_flg:
            logger.debug('start training tokenizer: %s' % tokenizer)
            tokenizer.train(output_files, vocab_size=30000)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            tokenizer.save(save_dir, name)
            logger.debug('saved tokenizer at %s' % save_dir)
        else:
            logger.debug('load trained tokenizer')

        # test decoding
        if debug:
            with open(output_files[-1], 'r') as f:
                for n, test in enumerate(f.read().split('<eos>')):
                    encoded = tokenizer.encode(test)
                    logger.debug(' - sample %i \n    * %s \n     * %s' % (n, test, str(encoded.tokens)))
                    if n > 10:
                        break

        # tokenize full corpus
        def convert_file(__file_path):
            logger.debug(' - converting file %s' % __file_path)
            with open(__file_path, 'r') as _f:
                token_ids = ' '.join([str(i) for i in tokenizer.encode(_f.read()).ids])
            __file_path = os.path.join(save_dir, __file_path.split('/')[-1]).replace('.txt', '.id.txt')
            with open(__file_path, 'w') as _f:
                _f.write(token_ids)
            logger.debug(' - saved at %s' % __file_path)
            return __file_path

        logger.info('tokenize corpus')
        save_dir = os.path.join(data_path, tokenizer)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        return [convert_file(_file) for _file in output_files]


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
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        seq = torch.tensor(sequence, dtype=torch.long, device=device)
        self.data_size = seq.size(0)

        n_batch = self.data_size // self.batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        seq = seq.narrow(0, 0, n_batch * self.batch_size)
        # Evenly divide the data across the bsz batches.
        self._data = seq.view(self.batch_size, -1).contiguous()

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
        if (self._index + 1) * self.num_steps + 1 > self._data.size(1):
            self._index = 0
            raise StopIteration
        x = self._data[:, self._index * self.num_steps:(self._index + 1) * self.num_steps].contiguous()
        y = self._data[:, self._index * self.num_steps + 1:(self._index + 1) * self.num_steps + 1].contiguous()
        self._index += 1
        return x, y