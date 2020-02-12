""" basic whitespace split tokenizer class """
import os
import json


__all__ = [
    "WhitespaceTokenizer"
]


class TokenObject:

    def __init__(self, tokens: list, ids: list):
        self.tokens = tokens
        self.ids = ids


class WhitespaceTokenizer:
    """ simple white space tokenzier """

    def __init__(self, vocab_path: str):
        self.__vocab_path = vocab_path
        if vocab_path is not None and os.path.exists(self.__vocab_path):
            self.__vocab = json.load(open(self.__vocab_path))
        else:
            self.__vocab = dict()

    def train(self, file_path_list: list, **kwargs):
        tokens = []
        for file_path in file_path_list:
            with open(file_path, 'r') as f:
                tokens += f.read().split()
        for t in tokens:
            if len(t) == 0:
                continue
            # if len(self.__vocab.keys()) > vocab_size:
            #     break
            if t not in self.__vocab.keys():
                self.__vocab[t] = len(self.__vocab.keys())

    def save(self, dir_to_save: str, name: str):
        with open(os.path.join(dir_to_save, '%s-vocab.json' % name), 'w') as f:
            json.dump(self.__vocab, f)

    def encode(self, text):
        tokens = text.split()
        ids = [self.__vocab[t] for t in tokens]
        return TokenObject(tokens, ids)

