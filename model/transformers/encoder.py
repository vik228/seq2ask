from __future__ import annotations

from encoder_layer import EncoderLayer
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Layer


class Encoder(Layer):

    def __init__(self,
                 input_vocab_size,
                 num_layers=4,
                 d_model=512,
                 n_heads=8,
                 dff=2048,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = Embedding(input_vocab_size, d_model, mask_zero=True)
        self.pos = SinePositionalEncoding()
        self.encoder_layers = [
            EncoderLayer(d_model, n_heads, dff, dropout)
            for _ in range(num_layers)
        ]
        self.dropout = Dropout(dropout)
