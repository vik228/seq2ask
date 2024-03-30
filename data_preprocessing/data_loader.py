from __future__ import annotations

import json
import pathlib
import random

import nltk
import requests

nltk.download("punkt")


class DataLoader():

    def __init__(self, version='2.0', **kwargs) -> None:
        assert version in ['1.1',
                           '2.0'], "Version must be either '1.1' or '2.0'"
        self.version = version
        self.sample_size = kwargs.get('sample_size', 10000)
        self.max_context_len = kwargs.get('max_context_len', 900)
        self.max_question_len = kwargs.get('max_question_len', 70)
        self.max_answer_len = kwargs.get('max_answer_len', 30)

    def download_squad(self):
        squad_base_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
        train_filename = f"train-v{self.version}.json"
        dev_filename = f"dev-v{self.version}.json"
        self.train_file = f"data/{train_filename}"
        self.dev_file = f"data/{dev_filename}"
        pathlib.Path("data/").mkdir(parents=True, exist_ok=True)
        pathlib.Path("data/").mkdir(parents=True, exist_ok=True)
        for filename, filepath in zip([train_filename, dev_filename],
                                      [self.train_file, self.dev_file]):
            url = f"{squad_base_url}{filename}"
            response = requests.get(url)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded {filename}")
            else:
                print(f"Failed to download {filename}")

    def load_squad_data(self, filename):
        with open(filename) as f:
            squad_dict = json.load(f)

        # Flatten the list of paragraphs for random sampling
        all_paragraphs = [
            paragraph for group in squad_dict['data']
            for paragraph in group['paragraphs']
        ]
        # Randomly sample paragraphs based on the sample_size
        if self.sample_size < len(all_paragraphs):
            sampled_paragraphs = random.sample(
                all_paragraphs, self.sample_size)
        else:
            sampled_paragraphs = all_paragraphs
        contexts = []
        questions = []
        answers = []
        for paragraph in sampled_paragraphs:
            context = paragraph['context']
            tokenized_context = nltk.word_tokenize(context)
            if len(tokenized_context) > self.max_context_len:
                continue
            for qa in paragraph['qas']:
                question = qa['question']
                tokenized_question = nltk.word_tokenize(question)
                if len(tokenized_question) > self.max_question_len:
                    continue
                for answer in qa['answers']:
                    tokenized_answer = nltk.word_tokenized(answer['text'])
                    if len(tokenized_answer) > self.max_answer_len:
                        continue
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer['text'])
        return contexts, questions, answers
