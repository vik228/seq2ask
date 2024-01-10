import json
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

class SquadPreprocessor:
    def __init__(self, squad_data_file, max_vocab_size=10000, max_seq_length=300):
        self.squad_data_file = squad_data_file
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.tokenizer = Tokenizer(num_words=max_vocab_size)
        self.contexts = []
        self.questions = []
        self.answers = []

    def load_and_parse_data(self):
        with open(self.squad_data_file, 'r') as file:
            squad_dict = json.load(file)
        for group in squad_dict['data']:
            for paragraph in group['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    question = qa['question']
                    for answer in qa['answers']:
                        self.contexts.append(context)
                        self.questions.append(question)
                        self.answers.append(answer)
    
    def tokenize_and_pad(self):
        combined_texts = self.contexts + self.questions
        self.tokenizer.fit_on_texts(combined_texts)
        context_sequences = self.tokenizer.texts_to_sequences(self.contexts)
        question_sequences = self.tokenizer.texts_to_sequences(self.questions)
        self.context_padded = pad_sequences(context_sequences, maxlen=self.max_seq_length, padding='post')
        self.question_padded = pad_sequences(question_sequences, maxlen=self.max_seq_length, padding='post')