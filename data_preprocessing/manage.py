from __future__ import annotations

from data_loader import DataLoader
from env import set_env_vars
from preprocessor import Preprocessor
from service import Service as DataPreprocessingService

if __name__ == '__main__':
    set_env_vars()
    max_vocab_size = 50000
    max_seq_len = 300
    data_loader = DataLoader()
    data_preprocessor = Preprocessor(max_vocab_size=max_vocab_size,
                                     max_seq_len=max_seq_len)
    service = DataPreprocessingService(data_loader=data_loader,
                                       data_preprocessor=data_preprocessor)
    service.prepare_squad_training_input(combine_context_and_questions=True)
