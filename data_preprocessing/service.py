class Service:
    def __init__(self, data_loader, data_preprocessor, mode='train') -> None:
        self.data_loader = data_loader
        self.data_preprocessor = data_preprocessor
        self.mode = mode
    
    def load_data(self):
        self.data_loader.download_squad()
        filename = self.data_loader.train_file if self.mode == 'train' else self.data_loader.dev_file
        return self.data_loader.load_squad_data(filename)

    def prepare_squad_training_input(self):
        contexts, questions, answers = self.load_data()
        return self.data_preprocessor.preprocess(contexts), self.data_preprocessor.preprocess(questions), self.data_preprocessor.preprocess(answers)
