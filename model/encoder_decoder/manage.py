from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from data_generator import DataGenerator
from env import bucket_name
from env import bucket_path_contexts_and_questions
from env import bucket_path_decoder_input
from env import bucket_path_decoder_output
from env import bucket_path_model
from env import set_env_vars
from optimizer import Optimizer
from sklearn.model_selection import train_test_split
from utils import download_blob

if __name__ == '__main__':
    set_env_vars()
    parser = argparse.ArgumentParser(
        description='Prepare data for training SV forecasting model.')
    parser.add_argument('--version',
                        type=str,
                        default='1.0',
                        help='Model Versioning.')
    parser.add_argument('--max_vocab_length',
                        type=str,
                        default=50000,
                        help='Maximum number of words in vocabulary.')

    parser.add_argument('--max_seq_len',
                        type=str,
                        default=300,
                        help='Maximum length of sequence.')

    parser.add_argument('--epochs',
                        type=str,
                        default=10,
                        help='Number of epochs to train.')
    directory = Path('data')
    directory.mkdir(parents=True, exist_ok=True)
    print("Directory '%s' created" % directory)
    args = parser.parse_args()
    max_vocab_length = int(args.max_vocab_length)
    max_seq_len = int(args.max_seq_len)
    lstm_units = 256
    embedding_dim = 128
    download_blob(bucket_name, bucket_path_contexts_and_questions,
                  'data/contexts_and_questions.npy')
    download_blob(bucket_name, bucket_path_decoder_input,
                  'data/decoder_input.npy')
    download_blob(bucket_name, bucket_path_decoder_output,
                  'data/decoder_output.npy')
    input_sequence = np.load('data/contexts_and_questions.npy')
    decoder_input_sequence = np.load('data/decoder_input.npy')
    decoder_output_sequence = np.load('data/decoder_output.npy')
    X_train, X_test, y_decoder_input_padded_train, y_decoder_input_padded_test, y_decoder_target_padded_train, y_decoder_target_padded_test = train_test_split(
        input_sequence,
        decoder_input_sequence,
        decoder_output_sequence,
        test_size=0.2,
        random_state=42)
    X_train, X_val, y_decoder_input_padded_train, y_decoder_input_padded_val, y_decoder_target_padded_train, y_decoder_target_padded_val = train_test_split(
        X_train,
        y_decoder_input_padded_train,
        y_decoder_target_padded_train,
        test_size=0.2,
        random_state=42)
    train_generator = DataGenerator(X_train, y_decoder_input_padded_train,
                                    y_decoder_target_padded_train)
    val_generator = DataGenerator(X_val, y_decoder_input_padded_val,
                                  y_decoder_target_padded_val)
    model_params = {
        'input_shape': max_seq_len,
        'layer_config': {
            'encoder': [{
                'type': 'embedding',
                'args': [max_vocab_length, embedding_dim],
                'kwargs': {
                    'input_length': max_seq_len,
                    'name': 'encoder_embedding'
                }
            }, {
                'type': 'lstm',
                'args': [lstm_units],
                'kwargs': {
                    'return_state': True,
                    'name': 'encoder_lstm'
                }
            }],
            'decoder': [
                {
                    'type': 'embedding',
                    'args': [max_vocab_length, embedding_dim],
                    'kwargs': {
                        'input_length': max_seq_len,
                        'name': 'decoder_embedding'
                    }
                },
                {
                    'type': 'lstm',
                    'args': [lstm_units],
                    'kwargs': {
                        'return_sequences': True,
                        'name': 'decoder_lstm',
                        'return_state': True
                    }
                },
            ],
            'output_layer': {
                'units': max_vocab_length,
                'activation': 'softmax',
                'name': 'output_layer'
            }
        },
        'optimizer': {
            'type': 'adam',
            'learning_rate': 0.001
        },
        'loss': 'sparse_categorical_crossentropy',
        'metrics': ['accuracy']
    }
    print("Model Params are %s" % model_params)
    training_params = {
        'train_generator': train_generator,
        'batch_size': 32,
        'epochs': int(args.epochs),
        'validation_generator': val_generator
    }
    print("Training Params are %s" % training_params)
    model_name = f"encoder_decoder_model_vocab_{max_vocab_length}_seq_{max_seq_len}_model_v{args.version}.h5"
    print("Model name is %s" % model_name)
    model_optimizer = Optimizer(model_params=model_params,
                                training_params=training_params,
                                model_bucket_path=bucket_path_model,
                                model_name=model_name)
    model_optimizer.build_and_train_model()
    model_optimizer.save_model(f"data/{model_name}")
