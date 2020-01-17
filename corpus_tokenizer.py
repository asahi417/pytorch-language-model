""" train tokenizer on dataset from torchtext with huggingface tokenizers
https://github.com/pytorch/text#datasets
https://github.com/huggingface/tokenizers/tree/master/bindings/python
"""
import tokenizers
import torchtext
import argparse
import os
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


def get_tokenizer(name, checkpoint_dir=None, checkpoint_name=None):
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

    valid_models = ['BPETokenizer', 'ByteLevelBPETokenizer', 'SentencePieceBPETokenizer', 'BertWordPieceTokenizer']
    if name == 'BPETokenizer':
        return tokenizers.BPETokenizer(vocab, merges), if_trained
    elif name == 'ByteLevelBPETokenizer':
        return tokenizers.ByteLevelBPETokenizer(vocab, merges), if_trained
    elif name == 'SentencePieceBPETokenizer':
        return tokenizers.SentencePieceBPETokenizer(vocab, merges), if_trained
    elif name == 'BertWordPieceTokenizer':
        return tokenizers.BertWordPieceTokenizer(vocab, merges), if_trained
    else:
        raise ValueError('unknown tokenizer %s, should be one of %s' % (name, str(valid_models)))


def get_data(name):
    data_field = torchtext.data.Field(sequential=True)
    if name == 'PennTreebank':
        torchtext.datasets.PennTreebank.splits(data_field, root='./data')
        file_path_train = './data/penn-treebank/ptb.train.txt'
        file_path_valid = './data/penn-treebank/ptb.valid.txt'
        file_path_test = './data/penn-treebank/ptb.test.txt'
        output_files = []
        for __file in [file_path_train, file_path_valid, file_path_test]:
            __file_output = __file.replace('.txt', '.eos.txt')
            output_files.append(__file_output)
            if os.path.exists(__file_output):
                continue
            with open(__file, 'r') as _f_r:
                with open(__file_output, 'w') as _f_w:
                    _f_w.write(_f_r.read().replace('\n', '<eos>'))
        return output_files
    else:
        raise ValueError('unknown data %s' % name)


def get_options():
    parser = argparse.ArgumentParser(description='Train tokenizer', formatter_class=argparse.RawTextHelpFormatter)
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-t', '--tokenizer', help='tokenizer', default='SentencePieceBPETokenizer', type=str, **share_param)
    parser.add_argument('-d', '--data', help='data', default='PennTreebank', type=str, **share_param)
    return parser.parse_args()


if __name__ == '__main__':
    logger = create_log()
    arguments = get_options()

    logger.info('tokenizer: %s' % arguments.tokenizer)
    save_dir = os.path.join("./vocab", arguments.tokenizer)
    tokenizer, if_trained_flg = get_tokenizer(arguments.tokenizer, checkpoint_dir=save_dir, checkpoint_name=arguments.data)

    logger.info('dataset: %s' % arguments.data)
    file_paths = get_data(arguments.data)

    # train tokenizer
    if not if_trained_flg:
        logger.info('start training tokenizer')
        tokenizer.train(file_paths)
        save_dir = os.path.join("./vocab", arguments.tokenizer)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        tokenizer.save(save_dir, arguments.data)
        logger.info('saved tokenizer at %s' % save_dir)

    else:
        logger.info('load trained tokenizer')

    # test encoding
    logger.info('test encoding')
    with open(file_paths[-1], 'r') as f:
        for n, test in enumerate(f.read().split('<eos>')):
            encoded = tokenizer.encode(test)
            logger.info(' sample %i \n * %s \n * %s' % (n, test, str(encoded.tokens)))
            if n > 10:
                break

    # tokenize full corpus
    logger.info('tokenize corpus')
    for _file in file_paths:
        logger.info(' - converting file %s' % _file)
        with open(_file, 'r') as f:
            token_ids = ' '.join([str(i) for i in tokenizer.encode(f.read()).ids])
            # token_ids_list = []
            # for n, text in enumerate(f.read().split('<eos>')):
            #     encoded_ids = tokenizer.encode(text).ids
            #     token_ids_list.append(' '.join([str(i) for i in encoded_ids]))
        # token_ids = '\n'.join(token_ids_list)
        _file = _file.replace('.txt', '.id.txt')
        with open(_file, 'w') as f:
            f.write(token_ids)
        logger.info(' - saved at %s' % _file)

    # save
    if not if_trained_flg:
        save_dir = os.path.join("./vocab", arguments.tokenizer)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        tokenizer.save(save_dir, arguments.data)
        logger.info('saved at %s' % save_dir)


