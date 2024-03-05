from __future__ import annotations


class Trainer:

    def __init__(self, model, training_params):
        self.model = model
        self.training_params = training_params
        self.history = None

    def train(self, X_train, y_train):
        batch_size = self.training_params.get('batch_size', 32)
        epochs = self.training_params['epochs']
        callbacks = self.training_params.get('callbacks', [])
        self.history = self.model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=0.2
            if 'validation_split' in self.training_params else None,
            validation_data=self.training_params.get('validation_data', None),
            shuffle=self.training_params.get('shuffle', True))

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
