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

    def prepare_squad_training_input(self, combine_context_and_questions=False):
        contexts, questions, answers = self.load_data()
        if combine_context_and_questions:
            inputs = [
                context + self.data_preprocessor.context_question_seperator +
                question for context, question in zip(contexts, questions)
            ]
            input_sequence = self.data_preprocessor.preprocess(inputs)
            output_sequence = self.data_preprocessor.preprocess(answers)
            return input_sequence, output_sequence
        return self.data_preprocessor.preprocess(
            contexts), self.data_preprocessor.preprocess(
                questions), self.data_preprocessor.preprocess(answers)
