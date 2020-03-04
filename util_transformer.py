""" pytorch transformer decoder implementation """

import math
import torch
import torch.nn as nn

__all__ = [
    "PositionalEmbedding",
    "Conv1D",
    "PointwiseFeedForward",
    "SelfMaskedAttention",
    "TransformerBlock",
    "TransformerDecoder"
]

EPS = 1
EPS_LAYER_NORM = 1


class PositionalEmbedding(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.register_buffer('inv_freq', 1 / (10000 ** (torch.arange(0.0, n_emb, 2.0) / n_emb)))

    def forward(self, pos_seq):
        """ positional embedding

         Parameter
        -----------
        pos_seq: 1-D tensor including sequence of relative position
        """

        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb


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
                 dropout_attention: float,
                 dropout_residual: float,
                 n_context: int,
                 n_positional_embedding: int):
        """ masked multi-heads self (causal) attention module with caching

         Parameter
        -------------
        n_embedding: int
            embedding dimension
        n_head: int
            number of attention head
        dropout_attention: float
        dropout_residual: float
        """
        super().__init__()
        assert n_embedding % n_head == 0
        self.linear_qkv = Conv1D(n_embedding, n_embedding * 3)  # 1d conv to get qkv once
        self.linear_heads = Conv1D(n_embedding, n_embedding)
        self.dropout_attention = nn.Dropout(dropout_attention)
        self.dropout_residual = nn.Dropout(dropout_residual)
        self.register_buffer(
            'mask',
            torch.tensor([[int(r <= c) for r in range(n_context)] for c in range(n_context)], dtype=torch.float))
        self.n_embedding = n_embedding
        self.n_head = n_head
        self.n_context = n_context
        if n_positional_embedding:
            self.linear_position = Conv1D(n_positional_embedding, n_embedding)
        else:
            self.linear_position = None

    def query_key_value(self, x, cached_key_value: list = None):
        """ get query/key/value vector for each head

         Parameter
        ------------
        x: tensor (batch, seq, dim)
        cached_key_value: list of two tensors (batch, n_head, dim / n_head, cached_seq), [cached_key, cached_value]

         Return
        ------------
        q: tensor (batch, self.n_head, seq, dim / self.n_head)
        v: tensor (batch, self.n_head, seq + cached_seq, dim / self.n_head)
        k: tensor (batch, self.n_head, dim / self.n_head, seq + cached_seq)

            * `cached_seq` is zero if cached_key_value is None.
        """

        def __split_into_heads(tensor):
            n_batch, n_seq, n_dim = tensor.size()
            assert n_dim == self.n_embedding
            tensor = tensor.view(n_batch, n_seq, self.n_head, int(self.n_embedding / self.n_head))
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
            return tensor

        qkv = self.linear_qkv(x)  # batch, seq, n_dim * 3
        q, k, v = torch.split(qkv, self.n_embedding, dim=-1)  # (batch, seq, n_dim) x 3
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

    def masked_attention_weight(self, q, k,
                                r_position_embedding=None,
                                r_content_bias=None,
                                r_position_bias=None):
        """ causal mask attention weight by lower triangular mask

        [[1., 0., 0., 0., 0.],
         [1., 1., 0., 0., 0.],
         [1., 1., 1., 0., 0.],
         [1., 1., 1., 1., 0.],
         [1., 1., 1., 1., 1.]]

         Parameter
        -----------
        q: tensor (batch, self.n_head, seq, dim / self.n_head)
        k: tensor (batch, self.n_head, dim / self.n_head, seq + cached_seq)
        r_position_embedding: tensor (seq + cached_seq, n_pos_dim)

         Return
        -----------
        att_weight: tensor (batch, head, seq, seq + cache)
            3rd axis is attended, and 4th is attending
        """
        att_weight = torch.matmul(q, k)
        batch, n_head, seq_attended, seq_attending = att_weight.size()
        cached_len = seq_attending - seq_attended
        assert self.n_head == n_head
        assert self.n_context == seq_attended
        if self.linear_position:
            assert r_position_embedding is not None
            assert r_content_bias is not None
            assert r_position_bias is not None
            assert cached_len >= 0
            # attended, attending, n_pos_emb
            rel_pos = torch.stack(
                [torch.stack([r_position_embedding[int(max(0, c - r)), :].contiguous()
                 for r in range(-cached_len, self.n_context)])
                 for c in range(self.n_context)])
            # attended, attending, n_emb
            rel_pos = self.linear_position(rel_pos)
            _dim = int(self.n_embedding / self.n_head)
            # 1, attended, attending, n_head, n_emb/n_head, 1
            rel_pos = rel_pos.view(1, self.n_context, seq_attending, self.n_head, _dim, 1)
            # 1, n_head, attended, attending, n_emb/n_head, 1
            rel_pos = rel_pos.permute(0, 3, 1, 2, 4, 5).contiguous()

            #######
            # (b) #
            #######
            # batch, n_head, attended, 1, 1, dim / self.n_head)
            _q = q.view(batch, self.n_head, self.n_context, 1, 1, _dim)
            # batch, n_head, attended, attending
            att_weight_new = torch.matmul(_q, rel_pos)
            assert att_weight_new.size(-1) == 1 and att_weight_new.size(-2) == 1
            att_weight_new = att_weight_new[:, :, :, :, 0, 0].contiguous()
            assert att_weight.shape == att_weight_new.shape
            att_weight = att_weight_new + att_weight

            #######
            # (c) #
            #######
            _r_content_bias = r_content_bias.view(1, self.n_head, 1, 1, 1, _dim)
            # batch, n_head, attended, attending
            att_weight_new = torch.matmul(_r_content_bias, rel_pos)
            assert att_weight_new.size(-1) == 1 and att_weight_new.size(-2) == 1 and att_weight_new.size(0) == 1
            att_weight_new = att_weight_new[:, :, :, :, 0, 0].contiguous()
            assert att_weight.shape[1:4] == att_weight_new.shape[1:4]
            att_weight = att_weight_new + att_weight

            #######
            # (d) #
            #######
            _r_position_bias = r_position_bias.view(1, self.n_head, 1, 1, 1, _dim)
            # batch, n_head, attended, attending
            att_weight_new = torch.matmul(_r_position_bias, rel_pos)
            assert att_weight_new.size(-1) == 1 and att_weight_new.size(-2) == 1 and att_weight_new.size(0) == 1
            att_weight_new = att_weight_new[:, :, :, :, 0, 0].contiguous()
            assert att_weight.shape[1:4] == att_weight_new.shape[1:4]
            att_weight = att_weight_new + att_weight

            # create mask for causal attention
            if cached_len == 0:
                mask = self.mask
            else:
                sub_mask = torch.ones((self.n_context, cached_len), device=self.mask.device, dtype=self.mask.dtype)
                mask = torch.cat([sub_mask, self.mask], dim=1)
        else:
            assert self.n_context == seq_attending and cached_len == 0
            mask = self.mask

        att_weight = self.masked_softmax(att_weight / (math.sqrt(q.size(-1)) + EPS), mask=mask, dim=-1)
        att_weight = self.dropout_attention(att_weight)
        return att_weight

    @staticmethod
    def masked_softmax(vec, mask, dim=1):
        """ softmax ignoring zero value """
        exps = torch.exp(vec.float())
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(dim, keepdim=True) + EPS
        return masked_exps / (masked_sums + EPS)

    def forward(self,
                x,
                cached_key_value: list=None,
                r_position_embedding=None,
                r_content_bias=None,
                r_position_bias=None):
        """ get attended context vector

         Parameter
        ------------
        x: tensor (batch, seq, dim), where the last row x[:, seq, :] is the newest token
        cached_key_value: list of two tensors [cached_key, cached_value],
            each has (batch, n_head, dim / n_head, cached_seq)
        pos_emb: tensor (1, seq + cached_seq, n_pos_dim)

         Return
        ------------
        context_vector: tensor (batch, seq, dim)
        (k, v): `key` tensor (batch, head, dim/head, seq + cache_size) and
                `value` tensor (batch, head, seq + cache_size, dim/head)
        """
        q, k, v = self.query_key_value(x, cached_key_value)
        # attention mask: batch, head, seq, seq + cache
        att_weight = self.masked_attention_weight(q, k, r_position_embedding, r_content_bias, r_position_bias)
        # print('att', att_weight)
        # batch, head, seq, dim/head
        context_vector = torch.matmul(att_weight, v)
        # batch, seq, dim/head, head
        context_vector = context_vector.permute(0, 2, 3, 1).contiguous()
        # batch, seq, dim
        context_vector = context_vector.view(context_vector.size(0), context_vector.size(1), -1)
        # merge head and residual dropout
        context_vector = self.linear_heads(context_vector)
        context_vector = self.dropout_residual(context_vector)
        print('cont', context_vector)
        return context_vector, (k, v)


class TransformerBlock(nn.Module):
    """ single Transformer Decoder Block """

    def __init__(self,
                 n_embedding: int,
                 n_state_ffn: int,
                 n_head: int,
                 dropout_residual: float,
                 dropout_attention: float,
                 n_context: int,
                 n_positional_embedding: int=None):
        """ single Transformer Decoder Block

         Parameter
        ------------
        n_embedding: int
            embedding dimension
        n_state_ffn: int
            intermediate state dimension
        n_head: int
            number of attention head
        dropout_residual: float
        dropout_attention: float
        """
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(n_embedding, eps=EPS_LAYER_NORM)
        self.layer_norm_2 = nn.LayerNorm(n_embedding, eps=EPS_LAYER_NORM)
        self.pointwise_ff = PointwiseFeedForward(n_embedding, n_state_ffn)
        self.self_attention = SelfMaskedAttention(n_embedding=n_embedding,
                                                  n_head=n_head,
                                                  dropout_attention=dropout_attention,
                                                  dropout_residual=dropout_residual,
                                                  n_context=n_context,
                                                  n_positional_embedding=n_positional_embedding)

    def forward(self,
                x,
                cached_key_value: list = None,
                r_position_embedding=None,
                r_content_bias=None,
                r_position_bias=None):
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
        c, (k, v) = self.self_attention(self.layer_norm_1(x),
                                        cached_key_value=cached_key_value,
                                        r_position_embedding=r_position_embedding,
                                        r_content_bias=r_content_bias,
                                        r_position_bias=r_position_bias)
        output = x + self.pointwise_ff(self.layer_norm_2(x + c))
        print('out', output)
        return output, (k, v)


class TransformerDecoder(nn.Module):
    """ Transformer Decoder """

    def __init__(self,
                 n_layer: int,
                 n_embedding: int,
                 n_state_ffn: int,
                 n_head: int,
                 dropout_embedding: float,
                 dropout_residual: float,
                 dropout_attention: float,
                 n_context: int,
                 n_positional_embedding: int = None):
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
        dropout_residual: float
        dropout_attention: float
        max_cache_size: int
            max cache size for key/value
        n_positional_embedding: int
            relative positional embedding dimension (no relative position encoding if None)
        """
        super().__init__()
        self.transformer_stack = nn.ModuleList([
            TransformerBlock(n_embedding=n_embedding,
                             n_state_ffn=n_state_ffn,
                             n_head=n_head,
                             dropout_residual=dropout_residual,
                             dropout_attention=dropout_attention,
                             n_context=n_context,
                             n_positional_embedding=n_positional_embedding)
            for _ in range(n_layer)
        ])
        self.input_dropout = nn.Dropout(dropout_embedding)
        self.layer_norm = nn.LayerNorm(n_embedding, eps=EPS_LAYER_NORM)  # eps=1e-5
        assert n_embedding % n_head == 0
        if n_positional_embedding and n_positional_embedding != 0:
            self.pos_emb = PositionalEmbedding(n_positional_embedding)
            self.r_c_bias = nn.Parameter(torch.zeros((n_head, int(n_embedding / n_head))))
            self.r_p_bias = nn.Parameter(torch.zeros((n_head, int(n_embedding / n_head))))
        else:
            self.r_c_bias = self.r_p_bias = self.pos_emb = None
        self.n_layer = n_layer
        self.n_context = n_context

    def forward(self, x, cached_key_value: list=None, max_cache_length: int=None):
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
        cached_length = cached_key_value[0][0].size(-1) if cached_key_value is not None else 0
        max_cache_length = min(cached_length, max_cache_length) if max_cache_length else cached_length

        if self.pos_emb:
            pos_seq = torch.arange(self.n_context + max_cache_length, device=x.device, dtype=x.dtype)
            pos_emb = self.pos_emb(pos_seq)  # (1, seq + cached - 1, dim)
            pos_emb = self.input_dropout(pos_emb)
        else:
            assert cached_length == 0
            pos_emb = None

        if cached_length == 0:
            cached_key_value = [None] * self.n_layer

        assert len(cached_key_value) == self.n_layer
        x = self.input_dropout(x)
        cached_key_value_new = []
        for transformer_block, cached_kv in zip(self.transformer_stack, cached_key_value):

            # limit cached context length
            if cached_kv is not None and max_cache_length < cached_length:
                k, v = cached_kv
                cached_kv = (k[:, :, :, -max_cache_length:].detach(), v[:, :, -max_cache_length:, :].detach())

            print('layer %i' % len(cached_key_value_new))
            x, (k, v) = transformer_block(x,
                                          cached_key_value=cached_kv,
                                          r_position_embedding=pos_emb,
                                          r_content_bias=self.r_c_bias,
                                          r_position_bias=self.r_p_bias)

            print()
            cached_key_value_new.append((k, v))

        exit()
        x = self.layer_norm(x)
        return x, cached_key_value_new

