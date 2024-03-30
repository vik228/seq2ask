from __future__ import annotations

import os


def set_env_vars():
    os.environ[
        'GOOGLE_APPLICATION_CREDENTIALS'] = 'tinygpt-408420-13a6caea8ade.json'
    os.environ['GOOGLE_CLOUD_PROJECT'] = 'tinygpt-408420'
    os.environ['GOOGLE_CLOUD_REGION'] = 'asia-south2'


bucket_name = 'seq-to-ask-data'
input_tokenizer_bucket_path = 'squad-data/processed/input_tokenizer'
output_tokenizer_bucket_path = 'squad-data/processed/output_tokenizer'
