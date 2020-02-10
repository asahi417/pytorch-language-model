""" pytorch GPT 2 implementation """

import math
import torch
import torch.nn as nn

__all__ = [
    "Conv1D",
    "PointwiseFeedForward",
    "SelfMaskedAttention",
    "TransformerBlock",
    "TransformerDecoder"
]


class PositionalEmbedding(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.register_buffer('inv_freq', 1 / (10000 ** (torch.arange(0.0, n_emb, 2.0) / n_emb)))

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


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
                 residual_dropout: float,
                 n_context: int):
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
        self.linear_qkv = Conv1D(n_embedding, n_embedding * 3)  # 1d conv to get qkv once
        self.linear_heads = Conv1D(n_embedding, n_embedding)  # 1d conv to get qkv once
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.residual_dropout = nn.Dropout(residual_dropout)
        self.register_buffer(
            'mask',
            torch.tensor([[int(r <= c) for r in range(n_context)] for c in range(n_context)], dtype=torch.float))
        self.__n_embedding = n_embedding
        self.__n_head = n_head
        self.__n_context = n_context

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

    def masked_attention_weight(self, q, k):
        """ causal mask attention weight by lower triangular mask

        [[1., 0., 0., 0., 0.],
         [1., 1., 0., 0., 0.],
         [1., 1., 1., 0., 0.],
         [1., 1., 1., 1., 0.],
         [1., 1., 1., 1., 1.]]

         Parameter
        -----------
        q: tensor (batch, self.__n_head, seq, dim / self.__n_head)
        k: tensor (batch, self.__n_head, dim / self.__n_head, seq + cached_seq)

         Return
        -----------
        att_weight: tensor (batch, head, seq, seq + cache)
            3rd axis is attended, and 4th is attending
        """
        att_weight = torch.matmul(q, k)
        batch, n_head, seq_attended, seq_attending = att_weight.size()
        cached_len = seq_attending - seq_attended
        assert cached_len > 0
        assert self.__n_context == seq_attended
        assert self.__n_head == n_head

        # apply mask to trainable part
        att_weight_no_cached = att_weight[:, :, :, cached_len:]
        att_weight_no_cached = self.masked_softmax(att_weight_no_cached / math.sqrt(q.size(-1)), dim=-1)
        att_weight_no_cached = self.attention_dropout(att_weight_no_cached)
        # concat all the cached part
        att_weight_cached = att_weight[:, :, :, :cached_len]
        att_weight_full = torch.cat([att_weight_cached, att_weight_no_cached], dim=-1)

        return att_weight_full

    def masked_softmax(self, vec, dim=1, epsilon=1e-5):
        """ softmax ignoring zero value """
        exps = torch.exp(vec.float())
        masked_exps = exps * self.mask.float()
        masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
        return masked_exps / masked_sums

    def forward(self, x, cached_key_value: list=None):
        """ get attended context vector

         Parameter
        ------------
        x: tensor (batch, seq, dim), where the last row x[:, seq, :] is the newest token
        cached_key_value: list of two tensors [cached_key, cached_value],
            each has (batch, n_head, dim / n_head, cached_seq)

         Return
        ------------
        context_vector: tensor (batch, seq, dim)
        (k, v): `key` tensor (batch, head, dim/head, seq + cache_size) and
                `value` tensor (batch, head, seq + cache_size, dim/head)
        """
        q, k, v = self.query_key_value(x, cached_key_value)
        # attention mask: batch, head, seq, seq + cache
        att_weight = self.masked_attention_weight(q, k)
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
                 attention_dropout: float,
                 n_context: int):
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
        self.layer_norm_1 = nn.LayerNorm(n_embedding)
        self.layer_norm_2 = nn.LayerNorm(n_embedding)
        self.pointwise_ff = PointwiseFeedForward(n_embedding, n_state_ffn)
        self.self_attention = SelfMaskedAttention(n_embedding=n_embedding,
                                                  n_head=n_head,
                                                  attention_dropout=attention_dropout,
                                                  residual_dropout=residual_dropout,
                                                  n_context=n_context)

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
        output = x + self.pointwise_ff(self.layer_norm_2(x + c))
        return output, (k, v)


class TransformerDecoder(nn.Module):
    """ Transformer Decoder in GPT2 """

    def __init__(self,
                 n_layer: int,
                 n_embedding: int,
                 n_state_ffn: int,
                 n_head: int,
                 residual_dropout: float,
                 attention_dropout: float,
                 n_context: int):
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
        self.transformer_stack = nn.ModuleList([
            TransformerBlock(n_embedding=n_embedding,
                             n_state_ffn=n_state_ffn,
                             n_head=n_head,
                             residual_dropout=residual_dropout,
                             attention_dropout=attention_dropout,
                             n_context=n_context)
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

            cached_key_value_new.append((k, v))

        x = self.layer_norm(x)
        return x, cached_key_value_new

