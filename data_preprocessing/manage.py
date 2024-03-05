from __future__ import annotations

from data_loader import DataLoader
from env import set_env_vars
from preprocessor import Preprocessor
from service import Service as DataPreprocessingService

if __name__ == '__main__':
    set_env_vars()
    data_loader = DataLoader()
    data_preprocessor = Preprocessor()
    service = DataPreprocessingService(data_loader=data_loader,
                                       data_preprocessor=data_preprocessor)
    input_sequence, output_sequence = service.prepare_squad_training_input(
        combine_context_and_questions=True)
