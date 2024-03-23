from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from data_generator import DataGenerator
from env import bucket_name
from env import bucket_path_model
from env import set_env_vars
from optimizer import Optimizer
from sklearn.model_selection import train_test_split
from utils import download_blob

if __name__ == '__main__':
    set_env_vars()
    parser = argparse.ArgumentParser(description='Model training for seq2ask.')
    cl_args = [
        ('--version', 'Model Versioning.'),
        ('--input_seq_path', 'Path to input sequence csv file.'),
        ('--decoder_input_path', 'Path to decoder input sequence csv file.'),
        ('--decoder_output_path', 'Path to decoder output sequence csv file.'),
        ('--model_params_bucket_path', 'Path to model params.'),
        ('--training_params_bucket_path', 'Path to training params.'),
    ]
    for arg in cl_args:
        parser.add_argument(
            arg[0],
            type=str,
            default=None,
            help=arg[1],
        )
    args = parser.parse_args()
    input_sequence = np.load(args.input_seq_path)
    decoder_input_sequence = np.load(args.decoder_input_path)
    decoder_output_sequence = np.load(args.decoder_output_path)
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
    Path.mkdir("data", exist_ok=True)
    download_blob(
        bucket_name=bucket_name,
        source_blob_name=args.model_params_bucket_path,
        destination_file_name="data/model_params.json",
    )
    download_blob(
        bucket_name=bucket_name,
        source_blob_name=args.training_params_bucket_path,
        destination_file_name="data/training_params.json",
    )
    model_params = json.load(open("data/model_params.json"))
    training_params = json.load(open("data/training_params.json"))
    logging.info("Model params are %s" % model_params)
    logging.info("Training params are %s" % training_params)
    training_params['train_generator'] = train_generator
    training_params['validation_generator'] = val_generator
    print("Training Params are %s" % training_params)
    model_name = f"encoder_decoder_model_v{args.version}.h5"
    print("Model name is %s" % model_name)
    model_optimizer = Optimizer(model_params=model_params,
                                training_params=training_params,
                                model_bucket_path=bucket_path_model,
                                model_name=model_name)
    model_optimizer.build_and_train_model()
    model_optimizer.save_model(f"data/{model_name}")
