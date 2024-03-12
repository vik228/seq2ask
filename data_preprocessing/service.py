from __future__ import annotations

import numpy as np
from env import bucket_path_contexts
from env import bucket_path_contexts_and_questions
from env import bucket_path_decoder_input
from env import bucket_path_decoder_output
from env import bucket_path_questions
from utils import upload_to_gcs


class Service:

    def __init__(self, data_loader, data_preprocessor, mode='train') -> None:
        self.data_loader = data_loader
        self.data_preprocessor = data_preprocessor
        self.mode = mode

    def load_data(self):
        self.data_loader.download_squad()
        filename = self.data_loader.train_file if self.mode == 'train' else self.data_loader.dev_file
        return self.data_loader.load_squad_data(filename)

    def prepare_squad_training_input(self, combine_context_and_questions):
        contexts, questions, answers = self.load_data()
        combined_data = contexts + questions + answers
        self.data_preprocessor.fit_on_texts(combined_data)
        if combine_context_and_questions:
            inputs = [
                context + self.data_preprocessor.context_question_seperator +
                question for context, question in zip(contexts, questions)
            ]
            input_sequence = self.data_preprocessor.preprocess(inputs)
            output_sequence = self.data_preprocessor.preprocess(answers,
                                                                is_answer=True)
            decoder_inputs = [seq[:-1] for seq in output_sequence]
            decoder_outputs = [seq[1:] for seq in output_sequence]
            np.save('data/input_sequence.npy', input_sequence)
            np.save('data/decoder_input_sequence.npy', decoder_inputs)
            np.save('data/decoder_output_sequence.npy', decoder_outputs)
            upload_to_gcs('data/input_sequence.npy',
                          bucket_path_contexts_and_questions)
            upload_to_gcs('data/decoder_input_sequence.npy',
                          bucket_path_decoder_input)
            upload_to_gcs('data/decoder_output_sequence.npy',
                          bucket_path_decoder_output)
        else:
            contexts_seq = self.data_preprocessor.preprocess(contexts)
            questions_seq = self.data_preprocessor.preprocess(questions)
            answers_seq = self.data_preprocessor.preprocess(answers)
            np.save('data/contexts_seq.npy', contexts_seq)
            np.save('data/questions_seq.npy', questions_seq)
            np.save('data/answers_seq.npy', answers_seq)
            upload_to_gcs('data/contexts_seq.npy', bucket_path_contexts)
            upload_to_gcs('data/questions_seq.npy', bucket_path_questions)
            #upload_to_gcs('data/answers_seq.npy', bucket_path_answers)
