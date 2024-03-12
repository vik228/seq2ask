from __future__ import annotations

from builder import Builder as EncoderDecoderBuilder
from trainer import Trainer as EncoderDecoderTrainer
from utils import upload_to_gcs


class Optimizer:

    def __init__(self, model_params, training_params, **kwargs):
        self.model_builder = EncoderDecoderBuilder(model_params)
        self.training_params = training_params
        self.model = None
        self.trainer = None
        self.model_bucket_path = kwargs.get('model_bucket_path')
        self.model_name = kwargs.get('model_name')

    def build_and_train_model(self):
        self.model = self.model_builder.build_model()
        self.trainer = EncoderDecoderTrainer(self.model, self.training_params)
        train_generator = self.training_params.get('train_generator')
        self.trainer.train(train_generator)

    def save_model(self, model_path):
        self.model.save(model_path)
        model_bucket_path = f"{self.model_bucket_path}/{self.model_name}.h5"
        upload_to_gcs(model_path, model_bucket_path)
