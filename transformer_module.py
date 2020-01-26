""" pytorch GPT 2 implementation """

import math
import torch
import torch.nn as nn

__all__ = [
    "Conv1D",
    "PointwiseFeedForward",
    "SelfMaskedAttention",
    "TransformerBlock",
    "TransformerDecoder",
    "BaseGPT2"
]


class Conv1D(nn.Module):
    """ 1d convolution """

    def __init__(self,
                 n_input: int,
                 n_output: int):
        """ 1d convolution

         Parameter
        -----------
        n_input: int
            input dimension
        n_output: int
            output dimension
        """
        super().__init__()
        self.__n_input = n_input
        self.__n_output = n_output
        self.linear = nn.Linear(self.__n_input, self.__n_output)
        self.linear.weight.data.normal_(std=0.02)

    def forward(self, x):
        """ module output

         Parameter
        -------------
        x: tensor (batch, sequence, input_dim)

         Return
        -------------
        x: tensor (batch, sequence, output_dim)
        """

        batch, seq, input_dim = x.size()
        assert input_dim == self.__n_input
        x = x.view(-1, self.__n_input)
        x = self.linear(x)
        x = x.view(batch, seq, self.__n_output)
        return x


class PointwiseFeedForward(nn.Module):
    """ point-wise feed forward network (1d conv -> gelu -> 1d conv)"""

    def __init__(self,
                 n_embedding: int,
                 n_state_ffn: int):
        """ point-wise feed forward network (1d conv -> gelu -> 1d conv)

         Parameter
        --------------
        n_embedding: int
            embedding dimension
        n_state_ffn: int
            intermediate state dimension
        """
        super().__init__()
        self.__n_state_ffn = n_state_ffn
        self.__n_embedding = n_embedding
        self.linear_1 = Conv1D(self.__n_embedding, self.__n_state_ffn)
        self.linear_2 = Conv1D(self.__n_state_ffn, self.__n_embedding)

    def forward(self, x):
        """ module output

         Parameter
        -------------
        x: tensor (batch, sequence, input_dim)

         Return
        -------------
        x: tensor (batch, sequence, input_dim)
        """

        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        return x

    @staticmethod
    def gelu(x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class SelfMaskedAttention(nn.Module):
    """ masked multiheads self (causal) attention module """

    def __init__(self,
                 n_embedding: int,
                 n_head: int,
                 attention_dropout: float,
                 residual_dropout: float):
        """ masked multi-heads self (causal) attention module with caching

         Parameter
        -------------
        n_embedding: int
            embedding dimension
        n_head: int
            number of attention head
        attention_dropout: float
        residual_dropout: float
        """
        super().__init__()
        assert n_embedding % n_head == 0
        self.__n_embedding = n_embedding
        self.__n_head = n_head
        self.linear_qkv = Conv1D(self.__n_embedding, self.__n_embedding * 3)  # 1d conv to get qkv once
        self.linear_heads = Conv1D(self.__n_embedding, self.__n_embedding)  # 1d conv to get qkv once
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.residual_dropout = nn.Dropout(residual_dropout)

    def query_key_value(self, x, cached_key_value: list=None):
        """ get query/key/value vector for each head

         Parameter
        ------------
        x: tensor (batch, seq, dim)
        cached_key_value: list of two tensors (batch, n_head, dim / n_head, cached_seq), [cached_key, cached_value]


         Return
        ------------
        q: tensor (batch, self.__n_head, seq, dim / self.__n_head)
        v: tensor (batch, self.__n_head, seq + cached_seq, dim / self.__n_head)
        k: tensor (batch, self.__n_head, dim / self.__n_head, seq + cached_seq)

            * `cached_seq` is zero if cached_key_value is None.
        """

        def __split_into_heads(tensor):
            n_batch, n_seq, n_dim = tensor.size()
            assert n_dim == self.__n_embedding
            tensor = tensor.view(n_batch, n_seq, self.__n_head, int(self.__n_embedding / self.__n_head))
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
            return tensor

        qkv = self.linear_qkv(x)  # batch, seq, n_dim * 3
        q, k, v = torch.split(qkv, self.__n_embedding, dim=-1)  # (batch, seq, n_dim) x 3
        q = __split_into_heads(q)
        v = __split_into_heads(v)
        k = __split_into_heads(k)
        k = k.permute(0, 1, 3, 2).contiguous()
        if cached_key_value is not None:
            cached_k, cached_v = cached_key_value
            assert list(k.size())[:-1] == list(cached_k.size())[:-1]
            assert list(v.permute(0, 1, 3, 2).size())[:-1] == list(cached_v.permute(0, 1, 3, 2).size())[:-1]
            v = torch.cat([cached_v, v], dim=2)
            k = torch.cat([cached_k, k], dim=3)
        return q, k, v

    def mask_attention_weight(self, att_weight):
        """ causal mask attention weight by lower triangular mask

        [[1., 0., 0., 0., 0.],
         [1., 1., 0., 0., 0.],
         [1., 1., 1., 0., 0.],
         [1., 1., 1., 1., 0.],
         [1., 1., 1., 1., 1.]]

         Parameter
        -----------
        att_weight: tensor (batch, head, seq, seq + cache)
            3rd axis is attended, and 4th is attending

         Return
        -----------
        att_weight: tensor (batch, head, seq, seq)
        """
        batch, n_head, seq_attended, seq_attending = att_weight.size()
        # print(att_weight.shape)
        cache_size = seq_attending - seq_attended
        assert cache_size >= 0
        assert n_head == self.__n_head
        mask = [[int(r + cache_size <= c) for r in range(seq_attending)] for c in range(seq_attended)]
        mask = torch.FloatTensor(mask)
        if att_weight.device.type == 'cuda':
            mask = mask.cuda()
        att_weight = mask * att_weight
        return att_weight

    def forward(self, x, cached_key_value: list=None):
        """ get attended context vector

         Parameter
        ------------
        x: tensor (batch, seq, dim), where the last row x[:, seq, :] is the newest token
        cached_key_value: list of two tensors (batch, n_head, dim / n_head, cached_seq), [cached_key, cached_value]

         Return
        ------------
        context_vector: tensor (batch, seq, dim)
        (k, v): `key` tensor (batch, head, dim/head, seq + cache_size) and
                `value` tensor (batch, head, seq + cache_size, dim/head)
        """
        q, k, v = self.query_key_value(x, cached_key_value)
        # print(v.shape)
        # attention mask: batch, head, seq, seq + cache
        att_weight = torch.matmul(q, k)
        att_weight = self.mask_attention_weight(att_weight)
        att_weight = torch.nn.functional.softmax(att_weight / math.sqrt(v.size(-1)), dim=-1)
        att_weight = self.attention_dropout(att_weight)
        # batch, head, seq, dim/head
        context_vector = torch.matmul(att_weight, v)
        # batch, seq, dim/head, head
        context_vector = context_vector.permute(0, 2, 3, 1).contiguous()
        # batch, seq, dim
        context_vector = context_vector.view(context_vector.size(0), context_vector.size(1), -1)
        # merge head and residual dropout
        context_vector = self.linear_heads(context_vector)
        context_vector = self.residual_dropout(context_vector)
        return context_vector, (k, v)


class TransformerBlock(nn.Module):
    """ single Transformer Decoder Block """

    def __init__(self,
                 n_embedding: int,
                 n_state_ffn: int,
                 n_head: int,
                 residual_dropout: float,
                 attention_dropout: float):
        """ single Transformer Decoder Block

         Parameter
        ------------
        n_embedding: int
            embedding dimension
        n_state_ffn: int
            intermediate state dimension
        n_head: int
            number of attention head
        residual_dropout: float
        attention_dropout: float
        """
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(n_embedding)  # eps=1e-5
        self.layer_norm_2 = nn.LayerNorm(n_embedding)  # eps=1e-5
        self.pointwise_ff = PointwiseFeedForward(n_embedding, n_state_ffn)
        self.self_attention = SelfMaskedAttention(n_embedding, n_head, attention_dropout, residual_dropout)

    def forward(self, x, cached_key_value: list=None):
        """ single transformer block

         Parameter
        ------------
        x: tensor (batch, seq, dim), where the last row x[:, seq, :] is the newest token
        cached_key_value: list of two tensors (batch, n_head, dim / n_head, cached_seq), [cached_key, cached_value]

         Return
        ------------
        x: tensor (batch, seq, dim)
        (k, v): `key` tensor (batch, head, dim/head, seq + cache_size) and
                `value` tensor (batch, head, seq + cache_size, dim/head)
        """
        c, (k, v) = self.self_attention(self.layer_norm_1(x), cached_key_value=cached_key_value)
        x += c
        x += self.pointwise_ff(self.layer_norm_2(x))
        return x, (k, v)


class TransformerDecoder(nn.Module):
    """ Transformer Decoder in GPT2 """

    def __init__(self,
                 n_layer: int,
                 n_embedding: int,
                 n_state_ffn: int,
                 n_head: int,
                 residual_dropout: float,
                 attention_dropout: float,
                 max_cache_size: int=None):
        """ Transformer Decoder

         Parameter
        ------------
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
        max_cache_size: int
            max cache size for key/value
        """
        super().__init__()
        self.__n_layer = n_layer
        self.__max_cache_size = max_cache_size
        self.transformer_stack = nn.ModuleList([
            TransformerBlock(n_embedding=n_embedding,
                             n_state_ffn=n_state_ffn,
                             n_head=n_head,
                             residual_dropout=residual_dropout,
                             attention_dropout=attention_dropout)
            for _ in range(self.__n_layer)
        ])
        self.layer_norm = nn.LayerNorm(n_embedding)  # eps=1e-5

    def forward(self, x, cached_key_value: list=None):
        """ transformer decoder output

         Parameter
        ------------
        x: tensor (batch, seq, dim), where the last row x[:, seq, :] is the newest token
        cached_key_value: cached key/value tensor

         Return
        ------------
        x: tensor (batch, seq, dim)
        cached_key_value_new: new cached_key_value
        """
        if cached_key_value is None:
            cached_key_value = [None] * self.__n_layer
        else:
            assert len(cached_key_value) == self.__n_layer

        cached_key_value_new = []
        for transformer_block, cached_kv in zip(self.transformer_stack, cached_key_value):
            x, (k, v) = transformer_block(x, cached_kv)
            # limit cached context length
            if self.__max_cache_size:
                cache_length = min(k.size(-1), self.__max_cache_size)
                k = k[:, :, :,  -cache_length:].detach()
                v = v[:, :, -cache_length:, :].detach()
                # print(k.shape, v.shape, x.shape)

            cached_key_value_new.append((k, v))

        x = self.layer_norm(x)
        return x, cached_key_value_new


class BaseGPT2(nn.Module):
    """ GPT2: transformer-based Language Model """

    def __init__(self,
                 n_layer: int,
                 n_embedding: int,
                 n_state_ffn: int,
                 n_head: int,
                 n_context: int,
                 max_cache_size: int,
                 residual_dropout: float,
                 attention_dropout: float,
                 embedding_dropout: float,
                 vocab_size: int,
                 initializer_range: float=0.02):
        """ GPT2: transformer-based Language Model

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
        max_cache_size: int
            max cache size for key/value
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
        if max_cache_size:
            assert max_cache_size >= 0
        else:
            max_cache_size = 0
        # position ids/embedding
        self.position_ids = torch.arange(0, n_context + max_cache_size, dtype=torch.long)
        self.position_embedding = nn.Embedding(n_context + max_cache_size, n_embedding)

        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.transformer_decoder = TransformerDecoder(
            n_layer=n_layer,
            n_embedding=n_embedding,
            n_state_ffn=n_state_ffn,
            n_head=n_head,
            residual_dropout=residual_dropout,
            attention_dropout=attention_dropout,
            max_cache_size=max_cache_size
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

    def forward(self, x, cached_key_value: list=None):
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

        if cached_key_value:
            start_position_id = cached_key_value[0][1].size(-2)
        else:
            start_position_id = 0

        # get embedding
        w_embedding = self.word_embedding(x)  # dropout embeddings
        position_ids = self.position_ids[start_position_id:start_position_id + x.size(-1)]
        if w_embedding.device.type == 'cuda':
            position_ids = position_ids.cuda()
        p_embedding = self.position_embedding(position_ids.unsqueeze(0))
        embedding = self.embedding_dropout(p_embedding + w_embedding)

        # transform
        logit, cached_key_value = self.transformer_decoder(embedding, cached_key_value)

        # get output
        batch, seq, dim = logit.size()
        logit = logit.view(batch * seq, dim)  # (batch, seq, dim) -> (batch * seq, dim)
        output = self.word_decoding(logit)  # (batch * seq, dim) -> (batch * seq, vocab)

        # get pred/prob
        pred = torch.max(output, dim=1)[1].view(batch, seq)
        prob = torch.nn.functional.softmax(output, dim=1).view(batch, seq, output.size(1))
        return (output, prob, pred), cached_key_value


if __name__ == '__main__':
    _batch, _seq, _dim = 10, 12, 100
    sample = torch.ones((_batch, _seq), dtype=torch.long)
    print('sample input:', sample.size())

    gpt = BaseGPT2(
        n_layer=12,
        n_embedding=_dim,
        n_state_ffn=200,
        n_head=int(_dim / 25),
        n_context=_seq,
        max_cache_size=_seq,
        residual_dropout=.1,
        attention_dropout=.1,
        embedding_dropout=.1,
        vocab_size=1000
    )
    (_output, _prob, _pred), kv = gpt(sample)
    print('outputs:', _output.shape, _prob.shape, _pred.shape)
    print(len(kv), len(kv[0]), kv[0][0].shape)
    (_output, _prob, _pred), kv = gpt(sample)
    print('outputs:', _output.shape, _prob.shape, _pred.shape)
    print(len(kv), len(kv[0]), kv[0][0].shape)
    (_output, _prob, _pred), kv = gpt(sample, kv)
    print('outputs:', _output.shape, _prob.shape, _pred.shape)
    print(len(kv), len(kv[0]), kv[0][0].shape)
