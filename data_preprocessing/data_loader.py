import json
import pathlib
import requests

class SquadDataLoader():

    def __init__(self, version='2.0') -> None:
        assert version in ['1.1', '2.0'], "Version must be either '1.1' or '2.0'"
        self.version = version

    def download_squad(self):
        squad_base_url = f"https://rajpurkar.github.io/SQuAD-explorer/dataset/"
        self.train_file = f"data/train-v{self.version}.json"
        self.dev_file = f"data/dev-v{self.version}.json"
        pathlib.Path(self.train_file).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.dev_file).mkdir(parents=True, exist_ok=True)
        for file in [self.train_file, self.dev_file]:
            url = f"{squad_base_url}{file}"
            response = requests.get(url)
            if response.status_code == 200:
                with open(file, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded {file}")
            else:
                print(f"Failed to download {file}")
    
    def load_squad_data(self, filename):
        with open(filename, 'r') as f:
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
                        answers.append(answer)
        return contexts, questions, answers
