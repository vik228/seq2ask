from __future__ import annotations

import numpy as np
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self,
                 input_padded,
                 decoder_input_padded,
                 decoder_target_padded,
                 batch_size=32):
        self.input_padded = input_padded
        self.decoder_input_padded = decoder_input_padded
        self.decoder_target_padded = decoder_target_padded
        self.indexes = np.arange(len(self.input_padded))
        self.batch_size = batch_size

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.input_padded) // self.batch_size

    def __getitem__(self, idx):
        encoder_input_seq = self.input_padded[idx * self.batch_size:(idx + 1) *
                                              self.batch_size]
        decoder_input_seq = self.decoder_input_padded[idx *
                                                      self.batch_size:(idx +
                                                                       1) *
                                                      self.batch_size]
        decoder_output_seq = self.decoder_target_padded[idx *
                                                        self.batch_size:(idx +
                                                                         1) *
                                                        self.batch_size]

        batch_x = [encoder_input_seq, decoder_input_seq]
        batch_y = decoder_output_seq
        return batch_x, batch_y
