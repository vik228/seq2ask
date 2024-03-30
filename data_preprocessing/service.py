from __future__ import annotations


class Service:

    def __init__(self,
                 data_loader,
                 input_data_preprocessor,
                 output_data_preprocessor,
                 mode='train') -> None:
        self.data_loader = data_loader
        self.input_data_preprocessor = input_data_preprocessor
        self.output_data_preprocessor = output_data_preprocessor
        self.mode = mode
        self.context_question_seperator = '<SEP>'
        self.answer_start = '<START>'
        self.answer_end = '<END>'

    def load_data(self):
        self.data_loader.download_squad()
        filename = self.data_loader.train_file if self.mode == 'train' else self.data_loader.dev_file
        return self.data_loader.load_squad_data(filename)

    def prepare_input(self, contexts, questions):
        all_input = contexts + questions
        self.input_data_preprocessor.fit_on_texts(all_input)
        inputs = [
            f"{context} {self.context_question_seperator} {question}"
            for context, question in zip(contexts, questions)
        ]
        input_sequence = self.input_data_preprocessor.preprocess(inputs)
        return input_sequence

    def prepare_output(self, answers):
        self.output_data_preprocessor.fit_on_texts(answers)
        outputs = [
            f"{self.answer_start} {answer} {self.answer_end}"
            for answer in answers
        ]
        output_sequences = self.output_data_preprocessor.preprocess(outputs)
        return output_sequences

    def prepare_squad_training_input(self):
        contexts, questions, answers = self.load_data()
        input_sequence = self.prepare_input(contexts, questions)
        output_sequences = self.prepare_output(answers)
        return input_sequence, output_sequences
