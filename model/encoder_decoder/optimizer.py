from __future__ import annotations

from builder import Builder as EncoderDecoderBuilder
from trainer import Trainer as EncoderDecoderTrainer
from utils import upload_to_gcs


class Optimizer:

    def __init__(self, model_params, training_params, **kwargs):
        self.model_builder = EncoderDecoderBuilder(model_params)
        self.training_params = training_params
        self.encoder_model = None
        self.decoder_model = None
        self.model = None
        self.trainer = None
        self.model_bucket_path = kwargs.get('model_bucket_path')
        self.model_name = kwargs.get('model_name')

    def build_and_train_model(self):
        self.model, self.encoder_model, self.decoder_model = self.model_builder.build_model(
        )
        self.trainer = EncoderDecoderTrainer(self.model, self.training_params)
        train_generator = self.training_params.get('train_generator')
        self.trainer.train(train_generator)

    def save_model(self, model_path, encoder_path, decoder_path):
        self.model.save(model_path)
        self.encoder_model.save(encoder_path)
        self.decoder_model.save(decoder_path)
        model_bucket_path = f"{self.model_bucket_path}{self.model_name}"
        encoder_model_path = f"{self.model_bucket_path}{encoder_path}"
        decoder_model_path = f"{self.model_bucket_path}{decoder_path}"
        upload_to_gcs(encoder_model_path, model_bucket_path)
        upload_to_gcs(decoder_model_path, model_bucket_path)
        upload_to_gcs(model_path, model_bucket_path)
