""" pytorch transformer decoder implementation

reference
- huggingface (torch) https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_gpt2.py#L329
- fairseq (torch)
    model https://github.com/pytorch/fairseq/blob/master/fairseq/models/transformer_lm.py
    transformer https://github.com/pytorch/fairseq/blob/master/fairseq/models/transformer.py
- https://github.com/huggingface/transformers/issues/2368
"""

# for model
import math
import torch
import torch.nn as nn


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
            tensor = tensor.view(n_batch, n_seq, self.__n_head, self.__n_embedding / self.__n_head)
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
            return tensor

        qkv = self.linear_qkv(x)  # batch, seq, n_dim * 3
        q, k, v = torch.split(qkv, self.__n_embedding, dim=-1)  # (batch, seq, n_dim) x 3
        q = __split_into_heads(q)
        v = __split_into_heads(v)
        k = __split_into_heads(k)
        if cached_key_value:
            cached_k, cached_v = cached_key_value
            assert list(k.size())[:-1] == list(cached_k.size())[:-1]
            assert list(v.size())[:-1] == list(cached_v.size())[:-1]
            v = torch.cat([cached_v, v], dim=2)
            k = torch.cat([cached_k, k], dim=3)
        k = k.permute(0, 1, 3, 2).contiguous()

        return q, k, v

    def mask_attention_weight(self, att_weight):
        """ causal mask attention weight by lower triangular mask

        [[1., 0., 0., 0., 0.],
         [1., 1., 0., 0., 0.],
         [1., 1., 1., 0., 0.],
         [1., 1., 1., 1., 0.],
         [1., 1., 1., 1., 1.]])

         Parameter
        -----------
        att_weight: tensor (batch, head, seq, seq + cache)
            3rd axis is attended, and 4th is attending

         Return
        -----------
        att_weight: tensor (batch, head, seq, seq)
        """
        batch, n_head, seq_attended, seq_attending = att_weight.size()
        cache_size = seq_attended - seq_attending
        assert cache_size >= 0
        assert n_head == self.__n_head
        mask = [[int(r + cache_size <= c) for r in range(seq_attending)] for c in range(seq_attended)]
        mask = torch.FloatTensor(mask)
        att_weight = mask * att_weight
        return att_weight

    def forward(self, x, cached_key_value: list=None):
        """ get scaled attended context vector

         Parameter
        ------------
        x: tensor (batch, seq, dim), where the last row x[:, seq, :] is the newest token
        cached_key_value: list of two tensors (batch, n_head, dim / n_head, cached_seq), [cached_key, cached_value]

         Return
        ------------
        context_vector: tensor (batch, seq, dim)
        """
        q, k, v = self.query_key_value(x)
        # attention mask: batch, head, seq, seq + cache
        att_weight = torch.matmul(q, k)
        att_weight = self.mask_attention_weight(att_weight)
        att_weight = torch.nn.functional.softmax(att_weight / math.sqrt(v.size(-1)), dim=-1)
        att_weight = self.attention_dropout(att_weight)
        # batch, head, seq, dim/head
        context_vector = att_weight * v
        # batch, seq, dim/head, head
        context_vector = context_vector.permute(0, 2, 3, 1)
        # batch, seq, dim
        context_vector = context_vector.view(context_vector.size(0), context_vector.size(1), -1)
        # merge head and residual dropout
        context_vector = self.linear_heads(context_vector)
        context_vector = self.residual_dropout(context_vector)
        return context_vector


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
        """
        x += self.self_attention(self.layer_norm_1(x), cached_key_value=cached_key_value)
        x += self.pointwise_ff(self.layer_norm_2(x))
        return x


class TransformerDecoder(nn.Module):
    """ Transformer Decoder """

    def __init__(self,
                 n_layer: int,
                 n_embedding: int,
                 n_state_ffn: int,
                 n_head: int,
                 residual_dropout: float,
                 attention_dropout: float):
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
        """
        super().__init__()
        self.__residual_dropout = residual_dropout
        self.__attention_dropout = attention_dropout
        self.transformer_stack = nn.ModuleList([
            TransformerBlock(n_embedding=n_embedding,
                             n_state_ffn=n_state_ffn,
                             n_head=n_head,
                             residual_dropout=residual_dropout,
                             attention_dropout=attention_dropout)
            for _ in range(n_layer)
        ])

    def forward(self, x, cached_key_value: list=None):
        """ transformer decoder output

         Parameter
        ------------
        x: tensor (batch, seq, dim), where the last row x[:, seq, :] is the newest token
        cached_key_value: list of two tensors (batch, n_head, dim / n_head, cached_seq), [cached_key, cached_value]

         Return
        ------------
        x: tensor (batch, seq, dim)
        """
        # TODO: how to feed cached_key_value?
        return self.transformer_stack(x, )


if __name__ == '__main__':
    gpt = TransformerDecoder(
        n_layer=12,
        n_embedding=1600,
        n_state_ffn=100,
        n_head=10,
        residual_dropout=.1,
        attention_dropout=.1
    )
    print(gpt)

