""" pytorch LSTM Language Model"""

import torch
import torch.nn as nn
from torch.autograd import Variable

__all__ = [
    "StackedLSTM"
]


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


class StackedLSTM(nn.Module):
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
        super().__init__()
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
        # (seq * batch, vocab) -> (batch, seq, vocab)
        output = output.view(emb.size(1), emb.size(0), self.__vocab_size)
        prob = prob.view(emb.size(1), emb.size(0), self.__vocab_size)
        # (seq * batch, vocab) -> (batch, seq,)
        pred = pred.view(emb.size(1), emb.size(0))
        return (output, prob, pred), new_hidden

    def repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""

        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

