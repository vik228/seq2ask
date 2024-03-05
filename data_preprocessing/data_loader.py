from __future__ import annotations

import json
import pathlib

import requests


class DataLoader():

    def __init__(self, version='2.0') -> None:
        assert version in ['1.1',
                           '2.0'], "Version must be either '1.1' or '2.0'"
        self.version = version

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
        contexts = []
        questions = []
        answers = []
        for group in squad_dict['data']:
            for paragraph in group['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    question = qa['question']
                    for answer in qa['answers']:
                        contexts.append(context)
                        questions.append(question)
                        answers.append(answer['text'])
        return contexts, questions, answers
