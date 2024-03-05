from __future__ import annotations

import os


def set_env_vars():
    os.environ[
        'GOOGLE_APPLICATION_CREDENTIALS'] = 'data_preprocessing/tinygpt-408420-13a6caea8ade.json'
    os.environ['GOOGLE_CLOUD_PROJECT'] = 'tinygpt-408420'
    os.environ['GOOGLE_CLOUD_REGION'] = 'asia-south2'


bucker_name = 'seq-to-ask-data'
bucket_path_contexts_and_questions = 'squad-data/processed/contexts_and_questions.npy'
bucket_path_answers = 'squad-data/processed/answers.npy'

bucket_path_questions = 'squad-data/processed/questions.npy'
bucket_path_contexts = 'squad-data/processed/contexts.npy'
