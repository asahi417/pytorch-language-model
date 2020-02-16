""" pytorch TransformerXL """
import torch
import torch.nn as nn
from util_transformer import TransformerDecoder

__all__ = ["TransformerXL"]


class TransformerXL(nn.Module):
    """ Transformer XL """

    def __init__(self,
                 n_layer: int,
                 n_embedding: int,
                 n_state_ffn: int,
                 n_head: int,
                 n_context: int,
                 residual_dropout: float,
                 attention_dropout: float,
                 embedding_dropout: float,
                 vocab_size: int,
                 n_positional_embedding: int,
                 initializer_range: float = 0.02):
        """ Transformer XL

         Parameter
        -----------------
        n_layer: int
            number of layer
        n_embedding: int
            embedding dimension
        n_state_ffn: int
            intermediate state dimension
        n_head: int
            number of attention head
        residual_dropout: float
        attention_dropout: float
        embedding_dropout: float
        n_context: int
            context length
        vocab_size: int
        initializer_range: float
        """
        super().__init__()

        # word embedding/decoding and position embedding
        self.word_embedding = nn.Embedding(vocab_size, n_embedding)
        # nn.Embedding(a, b).weight.shape -> (a, b), while nn.Linear(a, b) -> (b, a)
        self.word_decoding = nn.Linear(n_embedding, vocab_size, bias=False)
        self.word_decoding.weight = self.word_embedding.weight
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.transformer_decoder = TransformerDecoder(
            n_layer=n_layer,
            n_embedding=n_embedding,
            n_state_ffn=n_state_ffn,
            n_head=n_head,
            residual_dropout=residual_dropout,
            attention_dropout=attention_dropout,
            embedding_dropout=embedding_dropout,
            n_context=n_context,
            n_positional_embedding=n_positional_embedding
        )
        self.__initializer_range = initializer_range
        self.init_weight()

    def __init_weight(self, _module):
        self.initializer_range = 0.02
        if isinstance(_module, (nn.Linear, nn.Embedding)):
            _module.weight.data.normal_(mean=0.0, std=self.__initializer_range)
            if isinstance(_module, nn.Linear) and _module.bias is not None:
                _module.bias.data.zero_()
        elif isinstance(_module, nn.LayerNorm):
            _module.bias.data.zero_()
            _module.weight.data.fill_(1.0)

    def init_weight(self):
        self.apply(self.__init_weight)

    def forward(self, x, cached_key_value: list = None, max_cache_length: int=None):
        """ model output

         Parameter
        -------------
        x: token id batch tensor (batch, sequence_length)
        cached_key_value: cached key/value tensor

         Return
        -------------
        (output, prob, pred):
            output: raw output from Transformer decoder (sequence_length, batch, vocab size)
            prob: softmax activated output (sequence_length, batch, vocab size)
            pred: prediction (sequence_length, batch)
        cached_key_value: new cached_key_value
        """

        # get embedding
        w_embedding = self.word_embedding(x)  # dropout embeddings

        # transform
        logit, cached_key_value = self.transformer_decoder(w_embedding, cached_key_value, max_cache_length)

        # get output
        batch, seq, dim = logit.size()
        logit = logit.view(batch * seq, dim)  # (batch, seq, dim) -> (batch * seq, dim)
        output = self.word_decoding(logit).float()  # (batch * seq, dim) -> (batch * seq, vocab)

        # get pred/prob
        pred = torch.max(output, dim=1)[1].view(batch, seq)
        prob = torch.nn.functional.softmax(output, dim=1).view(batch, seq, output.size(1))
        output = output.view(batch, seq, output.size(1))
        return (output, prob, pred), cached_key_value


if __name__ == '__main__':
    _batch, _seq, _dim = 10, 12, 100
    sample = torch.ones((_batch, _seq), dtype=torch.long)
    print('sample input:', sample.size())

    gpt = TransformerXL(
        n_layer=2,
        n_embedding=_dim,
        n_state_ffn=200,
        n_head=int(_dim / 25),
        n_context=_seq,
        residual_dropout=.1,
        attention_dropout=.1,
        embedding_dropout=.1,
        vocab_size=1000,
        n_positional_embedding=10
    )
    print('\n * 1')
    (_output, _prob, _pred), kv = gpt(sample)
    print('outputs:', _output.shape, _prob.shape, _pred.shape)
    print(len(kv), len(kv[0]), kv[0][0].shape)

    print('\n * 2')
    (_output, _prob, _pred), kv = gpt(sample)
    print('outputs:', _output.shape, _prob.shape, _pred.shape)
    print(len(kv), len(kv[0]), kv[0][0].shape)

    print('\n * 3')
    (_output, _prob, _pred), kv = gpt(sample, kv)
    print('outputs:', _output.shape, _prob.shape, _pred.shape)
    print(len(kv), len(kv[0]), kv[0][0].shape)

    print('\n * 4')
    (_output, _prob, _pred), kv = gpt(sample, kv, max_cache_length=2)
    print('outputs:', _output.shape, _prob.shape, _pred.shape)
    print(len(kv), len(kv[0]), kv[0][0].shape)

