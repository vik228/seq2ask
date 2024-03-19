from __future__ import annotations

import argparse

from data_loader import DataLoader
from env import set_env_vars
from preprocessor import Preprocessor
from service import Service as DataPreprocessingService

if __name__ == '__main__':
    set_env_vars()
    parser = argparse.ArgumentParser(
        description='Prepare data for training SV forecasting model.')
    parser.add_argument('--max_vocab_length',
                        type=str,
                        default=50000,
                        help='Maximum number of words in vocabulary.')

    parser.add_argument('--max_seq_len',
                        type=str,
                        default=300,
                        help='Maximum length of sequence.')

    parser.add_argument('--sample_size',
                        type=str,
                        default=None,
                        help='Number of samples to use for training.')

    args = parser.parse_args()
    max_vocab_size = int(args.max_vocab_length)
    max_seq_len = int(args.max_seq_len)
    sample_size = args.sample_size
    if sample_size:
        data_loader = DataLoader(sample_size=int(sample_size))
    else:
        data_loader = DataLoader()
    data_preprocessor = Preprocessor(max_vocab_size=max_vocab_size,
                                     max_seq_len=max_seq_len)
    service = DataPreprocessingService(data_loader=data_loader,
                                       data_preprocessor=data_preprocessor)
    service.prepare_squad_training_input(combine_context_and_questions=True)
