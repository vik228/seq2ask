from __future__ import annotations

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


class Preprocessor():

    def __init__(self, max_vocab_size=10000, max_seq_len=300, **kwargs):
        self.oov_token = kwargs.get('oov_token', '<OOV>')
        self.context_question_seperator = kwargs.get(
            'context_question_seperator', '<SEP>')
        self.answer_start_token = kwargs.get('answer_start', '<START>')
        self.answer_end_token = kwargs.get('answer_end', '<END>')
        self.max_vocab_size = max_vocab_size
        self.max_seq_len = max_seq_len
        self.tokenizer = Tokenizer(num_words=max_vocab_size, oov_token='<OOV>')

    def pad_sequence(self, sequence):
        return pad_sequences([sequence],
                             maxlen=self.max_seq_len,
                             padding='post')[0]

    def text_to_sequence(self, text, is_answer=False):
        if is_answer:
            return self.tokenizer.texts_to_sequences(
                [self.answer_start_token + text + self.answer_end_token])[0]
        return self.tokenizer.texts_to_sequences([text])[0]

    def seq_to_text(self, seq):
        return self.tokenizer.sequences_to_texts([seq])[0]

    def fit_on_texts(self, texts):
        self.tokenizer.fit_on_texts(texts)

    def preprocess(self, texts, **kwargs):
        data_seq = [self.text_to_sequence(text, **kwargs) for text in texts]
        data_seq_padded = [self.pad_sequence(seq) for seq in data_seq]
        return data_seq_padded
