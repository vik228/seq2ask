from __future__ import annotations

from keras.layers import Attention
from keras.layers import Dense
from keras.layers import Layer
from keras.layers import Permute
from keras.layers import Reshape


class MultiHeadAttention(Layer):

    def __init__(self, d_model, n_heads, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % self.n_heads == 0
        self.depth = d_model // self.n_heads
        self.wq = Dense(d_model)
        self.split_reshape_query = Reshape(
            (-1, self.n_heads,
             self.depth))  # batch_size x seq_len x n_heads x depth
        self.permute_query = Permute(
            (2, 1, 3))  # batch_size x n_heads x seq_len x depth

        self.wk = Dense(d_model)
        self.split_reshape_key = Reshape(
            (-1, self.n_heads,
             self.depth))  # batch_size x seq_len x n_heads x depth
        self.permute_key = Permute(
            (2, 1, 3))  # batch_size x n_heads x seq_len x depth

        self.wv = Dense(d_model)
        self.split_reshape_value = Reshape(
            (-1, self.n_heads,
             self.depth))  # batch_size x seq_len x n_heads x depth
        self.permute_value = Permute(
            (2, 1, 3))  # batch_size x n_heads x seq_len x depth

        self.attention = Attention(causal=True,
                                   dropout=kwargs.get('dropout', 0.1))
        self.join_permute_attention = Permute(2, 1, 3)
        self.join_reshape_attention = Reshape((-1, self.d_model))

        self.dense = Dense(d_model)  # batch_size x seq_len x d_model

    def call(self, inputs, mask=None, training=None):
        q = inputs[0]
        k = inputs[1]
        v = inputs[2]

        q = self.wq(q)
        q = self.split_reshape_query(q)
        q = self.permute_query(q)

        k = self.wk(k)
        k = self.split_reshape_key(k)
        k = self.permute_key(k)

        v = self.wv(v)
        v = self.split_reshape_value(v)
        v = self.permute_value(v)
        if mask:
            if mask[0]:
                mask[0] = Reshape(-1, 1)(mask[0])
                mask[0] = Permute(2, 1)(mask[0])
            if mask[1]:
                mask[1] = Reshape(-1, 1)(mask[0])
                mask[1] = Permute(2, 1)(mask[1])
        output = self.attention([q, k, v], mask=mask)
        output = self.join_permute_attention(output)
        output = self.join_reshape_attention(output)
        output = self.dense(output)
        return output
