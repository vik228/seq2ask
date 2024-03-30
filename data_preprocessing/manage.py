from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from data_loader import DataLoader
from env import input_tokenizer_bucket_path
from env import output_tokenizer_bucket_path
from env import set_env_vars
from preprocessor import Preprocessor
from service import Service as DataPreprocessingService
from utils import upload_to_gcs

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    set_env_vars()
    cli_args = [
        ('--input_max_vocab_length',
         'Maximum number of words in input vocabulary.'),
        ('--output_max_vocab_length',
         'Maximum number of words in output vocabulary.'),
        ('--input_max_seq_len', 'Maximum length of sequence of input.'),
        ('--output_max_seq_len', 'Maximum length of sequence of output.'),
        ('--max_context_len', 'Maximum length of context.'),
        ('--max_question_len', 'Maximum length of question.'),
        ('--max_answer_len', 'Maximum length of answer.'),
        ('--sample_size', 'Number of samples to use for training.'),
        ('--input_seq_path', 'Path to input sequence numpy file.'),
        ('--output_seq_path', 'Path to output sequence numpy file.'),
    ]
    parser = argparse.ArgumentParser(
        description='Prepare data for training seq2ask model.')
    for arg in cli_args:
        parser.add_argument(
            arg[0],
            type=str,
            default=None,
            help=arg[1],
        )
    Path("data").mkdir(exist_ok=True)
    args = parser.parse_args()
    input_max_vocab_length = int(args.input_max_vocab_length)
    output_max_vocab_length = int(args.output_max_vocab_length)
    input_max_seq_len = int(args.input_max_seq_len)
    output_max_seq_len = int(args.output_max_seq_len)
    sample_size = args.sample_size
    data_loader = DataLoader(sample_size=int(sample_size),
                             max_context_len=int(args.max_context_len),
                             max_question_len=int(args.max_question_len),
                             max_answer_len=int(args.max_answer_len))
    input_data_preprocessor = Preprocessor(
        max_vocab_size=input_max_vocab_length, max_seq_len=input_max_seq_len)
    output_data_preprocessor = Preprocessor(
        max_vocab_size=output_max_vocab_length, max_seq_len=output_max_seq_len)

    service = DataPreprocessingService(
        data_loader=data_loader,
        input_data_preprocessor=input_data_preprocessor,
        output_data_preprocessor=output_data_preprocessor)
    input_seq, output_seq = service.prepare_squad_training_input()
    logger.info(f"Saving input_seq to {args.input_seq_path}")
    logger.info(f"Saving output_seq to {args.output_seq_path}")
    np.save(args.input_seq_path, input_seq)
    np.save(args.decoder_output_path, output_seq)
    input_tokenizer = input_data_preprocessor.tokenizer.to_json()
    output_tokenizer = output_data_preprocessor.tokenizer.to_json()
    with open("data/input_tokenizer.json", "w") as f:
        f.write(json.dumps(input_tokenizer, ensure_ascii=False, indent=4))
    with open("data/output_tokenizer.json", "w") as f:
        f.write(json.dumps(output_tokenizer, ensure_ascii=False, indent=4))
    upload_to_gcs("data/input_tokenizer.json", input_tokenizer_bucket_path)
    upload_to_gcs("data/output_tokenizer.json", output_tokenizer_bucket_path)
