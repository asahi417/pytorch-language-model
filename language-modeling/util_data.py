""" Data process modules """
import os
import torch
import torchtext
import tokenizers
import zipfile
import argparse
from util import create_log
from util_base_tokenizer import WhitespaceTokenizer


__all__ = [
    "get_data",
    "get_tokenizer",
    "BatchFeeder",
    "VALID_DATA_LIST",
    "VALID_TOKENIZER_LIST"
]


VALID_DATA_LIST = ['PennTreebank', 'WikiText103', 'enwiki8']
VALID_TOKENIZER_LIST = ['BPETokenizer', 'ByteLevelBPETokenizer', 'SentencePieceBPETokenizer', 'BertWordPieceTokenizer',
                        'WhitespaceTokenizer']


def enwiki8_dataset(data_dir, num_test_chars: int = 5000000):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    output_list = \
        [os.path.join(data_dir, 'train.txt'), os.path.join(data_dir, 'valid.txt'), os.path.join(data_dir, 'test.txt')]
    if os.path.exists(os.path.join(data_dir, 'train.txt')) and \
        os.path.exists(os.path.join(data_dir, 'valid.txt')) and \
            os.path.exists(os.path.join(data_dir, 'test.txt')):
        return output_list

    os.system("wget --continue http://mattmahoney.net/dc/enwik8.zip")
    os.system("mv ./enwik8.zip %s/" % data_dir)
    if not os.path.exists(os.path.join(data_dir, 'enwik8.zip')):
        raise ValueError('download enwik8 failed, please download it from `http://mattmahoney.net/dc/enwik8.zip` '
                         'and put it at `%s`' % os.path.join(data_dir, 'enwik8.zip'))
    data = zipfile.ZipFile(os.path.join(data_dir, 'enwik8.zip')).read('enwik8')
    train_data = data[: -2 * num_test_chars]
    valid_data = data[-2 * num_test_chars: -num_test_chars]
    test_data = data[-num_test_chars:]
    for fn, part in [('train.txt', train_data), ('valid.txt', valid_data), ('test.txt', test_data)]:
        open(os.path.join(data_dir, fn), 'w').write(' '.join([str(c) if c != ord('\n') else '\n' for c in part]))
    return output_list


def get_tokenizer(name: str, checkpoint_dir: str = None, checkpoint_name: str = None):
    """ get tokenizer instance"""
    if_trained = False
    if checkpoint_dir is not None and checkpoint_name is not None:
        merges = '%s-merges.txt' % os.path.join(checkpoint_dir, checkpoint_name)
        vocab = '%s-vocab.json' % os.path.join(checkpoint_dir, checkpoint_name)
        if name in ['BPETokenizer', 'ByteLevelBPETokenizer', 'SentencePieceBPETokenizer']:
            if_trained = os.path.exists(merges) and os.path.exists(vocab)
        else:
            if_trained = os.path.exists(vocab)
        if not if_trained:
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
    elif name == 'WhitespaceTokenizer':
        return WhitespaceTokenizer(vocab), if_trained
    else:
        raise ValueError('unknown tokenizer %s' % name)


def get_data(name,
             tokenizer_name: str = 'SentencePieceBPETokenizer',
             data_directory: str = './data',
             vocab_directory: str = './vocab',
             eos_symbol: str = '<eos>',
             special_tokens: list = None,
             debug: bool = False,
             vocab_size: int = 30000):
    """ Get file path to tokenized benchmark data

     Parameter
    ------------
    name: dataset name
    tokenizer: tokenizer name
    data_directory: path to cache data
    vocab_directory: path to save tokenizer model

     Return
    ------------
    list of token_id list (train/valid/test)
    """
    logger = create_log()
    if special_tokens is not None:
        special_tokens += eos_symbol
    else:
        special_tokens = [eos_symbol]
    if '<unk>' not in special_tokens:
        special_tokens += ['<unk>']

    # fetch benchmark data
    def convert_eos(__file_path):
        __file_output = '.'.join(__file_path.split('.')[:-1]) + '.eos.txt'
        if not os.path.exists(__file_output):
            with open(__file_path, 'r') as _f_r:
                with open(__file_output, 'w') as _f_w:
                    for line in _f_r:
                        _f_w.write(' '.join(line.strip().split() + [eos_symbol]))
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
            convert_eos(os.path.join(data_path, _f)) for _f in ['wiki.train.tokens', 'wiki.valid.tokens', 'wiki.test.tokens']]
    elif name == 'enwiki8':
        data_path = os.path.join(data_directory, 'enwik8')
        output_files = enwiki8_dataset(data_dir=data_path)
    else:
        raise ValueError('unknown data %s' % name)
    logger.debug('data %s has been downloaded' % name)

    # train tokenizer
    save_dir = os.path.join(vocab_directory, tokenizer_name)
    tokenizer, if_trained_flg = get_tokenizer(tokenizer_name, checkpoint_dir=save_dir, checkpoint_name=name)
    if not if_trained_flg:
        logger.debug('start training tokenizer: %s' % tokenizer_name)
        if tokenizer_name == 'WhitespaceTokenizer':
            tokenizer.train(output_files)
        else:
            tokenizer.train(output_files, vocab_size=vocab_size, special_tokens=special_tokens)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        tokenizer.save(save_dir, name)
        logger.debug('saved tokenizer at %s' % save_dir)
    else:
        logger.debug('load trained tokenizer')

    # test decoding
    if debug:
        encoded = tokenizer.encode(open(output_files[-1], 'r').read()[:100])
        logger.debug(' - sample decoding \n     * %s' % str(encoded.tokens))

    # tokenize full corpus
    def convert_file(__file_path):
        __file_path_output = os.path.join(save_dir, __file_path.split('/')[-1]).replace('.txt', '.id.txt')
        if os.path.exists(__file_path_output):
            return [int(i) for i in open(__file_path_output, 'r').read().split()]

        logger.debug(' - converting file %s' % __file_path)
        token_ids = [int(i) for i in tokenizer.encode(open(__file_path, 'r').read()).ids]
        open(__file_path_output, 'w').write(' '.join([str(i) for i in token_ids]))
        logger.debug(' - saved at %s' % __file_path_output)
        return token_ids

    logger.info('tokenize corpus')
    save_dir = os.path.join(data_path, tokenizer_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    output_files_ids = []
    for _file in output_files:
        output_files_ids.append(convert_file(_file))
    return output_files_ids


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
        raw_sequence = torch.tensor(sequence, dtype=torch.long)
        self.data_size = raw_sequence.size(0)
        n_batch = self.data_size // self.batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        trimed_seq = raw_sequence.narrow(0, 0, n_batch * self.batch_size)
        # Evenly divide the data across the bsz batches.
        self.data = trimed_seq.view(self.batch_size, -1).contiguous()
        self.raw_sequnece = raw_sequence

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
        if (self._index + 1) * self.num_steps + 1 > self.data.size(1):
            self._index = 0
            raise StopIteration
        x = self.data[:, self._index * self.num_steps:(self._index + 1) * self.num_steps].contiguous()
        y = self.data[:, self._index * self.num_steps + 1:(self._index + 1) * self.num_steps + 1].contiguous()
        self._index += 1
        return x, y


def get_options():
    parser = argparse.ArgumentParser(description='Train language model', formatter_class=argparse.RawTextHelpFormatter)
    _p = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-d', '--data', help='data: %s' % str(VALID_DATA_LIST), default='PennTreebank', type=str, **_p)
    parser.add_argument('-t', '--tokenizer', help='tokenizer: %s' % str(VALID_TOKENIZER_LIST), default='SentencePieceBPETokenizer', type=str, **_p)
    return parser.parse_args()


if __name__ == '__main__':
    arguments = get_options()
    # vocab size can be applied for tokenizer except Whitespace
    get_data(arguments.data, arguments.tokenizer, vocab_size=10000)
