from __future__ import annotations

import numpy as np
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, input_seq, output_seq, batch_size=32):
        self.input_seq = input_seq
        self.output_seq = output_seq
        self.batch_size = batch_size
        self.decoder_input_padded = np.array(
            [seq[:-1] for seq in self.output_seq])
        self.decoder_target_padded = np.array(
            [seq[1:] for seq in self.output_seq])
        self.indexes = np.arange(len(self.input_seq))
        self.batch_size = batch_size

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.input_seq) // self.batch_size

    def __getitem__(self, idx):
        encoder_input_seq = self.input_seq[idx * self.batch_size:(idx + 1) *
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
