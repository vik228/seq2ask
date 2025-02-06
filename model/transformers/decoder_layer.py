from __future__ import annotations

from keras.layers import Add
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Layer
from keras.layers import LayerNormalization
from keras.layers import MultiHeadAttention


class DecoderLayer(Layer):

    def __init__(self, d_model, n_heads, dff, dropout=0.1, activation='relu'):
        super().__init__()
        self.masked_multi_head_attention = MultiHeadAttention(d_model, n_heads)
        self.dropout_masked_attention = Dropout(dropout)
        self.add_masked_attention = Add()
        self.layer_norm_masked_attention = LayerNormalization(epsilon=1e-6)

        self.multi_head_attention = MultiHeadAttention(d_model, n_heads)
        self.dropout_attention = Dropout(dropout)
        self.add_attention = Add()
        self.layer_norm_attention = LayerNormalization(epsilon=1e-6)

        self.ff_layer = Dense(dff, activation=activation)
        self.ff_layer_dense = Dense(d_model)
        self.dropout_dense = Dense(dropout)
        self.add_dense = Add()
        self.layer_norm_dense = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, mask=None, training=None):
        masked_attention = self.masked_multi_head_attention(
            [inputs, inputs, inputs], mask=[mask, mask])
        masked_attention = self.dropout_masked_attention(masked_attention,
                                                         training=training)
        x = self.add_masked_attention([inputs, masked_attention])
        x = self.layer_norm_masked_attention(x)

        attention = self.multi_head_attention([x, x, x], mask=[mask, mask])
        attention = self.dropout_attention(attention, training=training)
        x = self.add_attention([x, attention])
        x = self.layer_norm_attention(x)

        ff = self.ff_layer(x)
        ff = self.ff_layer_dense(ff)
        ff = self.dropout_dense(ff, training=training)
        x = self.add_dense([x, ff])
        x = self.layer_norm_dense(x)
        return x
