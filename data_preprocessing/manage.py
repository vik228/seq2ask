from __future__ import annotations

import argparse
import logging

import numpy as np
from data_loader import DataLoader
from env import set_env_vars
from preprocessor import Preprocessor
from service import Service as DataPreprocessingService

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    set_env_vars()
    cli_args = [
        ('--max_vocab_length', 'Maximum number of words in vocabulary.'),
        ('--max_seq_len', 'Maximum length of sequence.'),
        ('--sample_size', 'Number of samples to use for training.'),
        ('--input_seq_path', 'Path to input sequence csv file.'),
        ('--decoder_input_path', 'Path to decoder input sequence csv file.'),
        ('--decoder_output_path', 'Path to decoder output sequence csv file.'),
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
    input_seq, decoder_iputs, decoder_outputs = service.prepare_squad_training_input(
        combine_context_and_questions=True)
    logger.info(f"Saving input_seq to {args.input_seq_path}")
    logger.info(f"Saving decoder_inputs to {args.decoder_input_path}")
    logger.info(f"Saving decoder_outputs to {args.decoder_output_path}")
    np.save(args.input_seq_path, input_seq)
    np.save(args.decoder_input_path, decoder_iputs)
    np.save(args.decoder_output_path, decoder_outputs)
