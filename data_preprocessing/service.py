from __future__ import annotations


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
            return input_sequence, decoder_inputs, decoder_outputs
        else:
            contexts_seq = self.data_preprocessor.preprocess(contexts)
            questions_seq = self.data_preprocessor.preprocess(questions)
            answers_seq = self.data_preprocessor.preprocess(answers)
            return contexts_seq, questions_seq, answers_seq
