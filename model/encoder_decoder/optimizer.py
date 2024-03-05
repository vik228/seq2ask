from __future__ import annotations

from builder import Builder as EncoderDecoderBuilder
from trainer import Trainer as EncoderDecoderTrainer


class Optimizer:

    def __init__(self, model_params, training_params, **kwargs):
        self.model_builder = EncoderDecoderBuilder(model_params)
        self.training_params = training_params

    def build_and_train_model(self):
        self.model = self.model_builder.build_model()
        self.trainer = EncoderDecoderTrainer(self.model, self.training_params)
        X_train, y_train = self.training_params.get('train_data')
        self.trainer.train(X_train=X_train, y_train=y_train)
