""" Will not be used annymore
train tokenizer on dataset from torchtext with huggingface tokenizers
https://github.com/pytorch/text#datasets
https://github.com/huggingface/tokenizers/tree/master/bindings/python

https://github.com/Smerity/sha-rnn/blob/master/getdata.sh
https://torchtext.readthedocs.io/en/latest/datasets.html#wikitext103
https://github.com/Smerity/sha-rnn/blob/master/data.py#L27
"""
import argparse
import os
from util import create_log
from data import


def get_options():
    parser = argparse.ArgumentParser(description='Train tokenizer', formatter_class=argparse.RawTextHelpFormatter)
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-t', '--tokenizer', help='tokenizer', default='SentencePieceBPETokenizer', type=str, **share_param)
    parser.add_argument('-d', '--data', help='data', default='PennTreebank', type=str, **share_param)
    return parser.parse_args()


if __name__ == '__main__':
    _logger = create_log()
    arguments = get_options()

    _logger.info('tokenizer: %s' % arguments.tokenizer)
    save_dir = os.path.join("./vocab", arguments.tokenizer)
    tokenizer, if_trained_flg = get_tokenizer(arguments.tokenizer, checkpoint_dir=save_dir, checkpoint_name=arguments.data)

    _logger.info('dataset: %s' % arguments.data)
    file_paths = get_data(arguments.data)

    # train tokenizer
    if not if_trained_flg:
        _logger.info('start training tokenizer')
        tokenizer.train(file_paths, vocab_size=30000)
        save_dir = os.path.join("./vocab", arguments.tokenizer)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        tokenizer.save(save_dir, arguments.data)
        # vocab_size = len(json.load(open('./vocab/%s/%s-vocab.json' % (arguments.tokenizer, arguments.data))))
        _logger.info('saved tokenizer at %s' % save_dir)
        # _logger.info('vocab: %i' % vocab_size)

    else:
        _logger.info('load trained tokenizer')

    # test encoding
    _logger.info('test encoding')
    with open(file_paths[-1], 'r') as f:
        for n, test in enumerate(f.read().split('<eos>')):
            encoded = tokenizer.encode(test)
            _logger.info(' sample %i \n * %s \n * %s' % (n, test, str(encoded.tokens)))
            if n > 10:
                break

    # tokenize full corpus
    _logger.info('tokenize corpus')
    for _file in file_paths:
        _logger.info(' - converting file %s' % _file)
        with open(_file, 'r') as f:
            token_ids = ' '.join([str(i) for i in tokenizer.encode(f.read()).ids])
        _file = _file.replace('.txt', '.id.txt')
        with open(_file, 'w') as f:
            f.write(token_ids)
        _logger.info(' - saved at %s' % _file)

    # save
    if not if_trained_flg:
        save_dir = os.path.join("./vocab", arguments.tokenizer)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        tokenizer.save(save_dir, arguments.data)
        _logger.info('saved at %s' % save_dir)


