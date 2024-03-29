from __future__ import annotations

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD


class Builder:

    def __init__(self, model_params):
        self.model_params = model_params
        # Define a mapping from layer type names to actual layer classes
        self.layer_mapping = {
            'embedding': Embedding,
            'lstm': LSTM,
            'dense': Dense,
            'bidirectional': Bidirectional,
            'dropout': Dropout,
            'batch_normalization': BatchNormalization,
            'layer_normalization': LayerNormalization,
        }

    def update_params(self, new_model_params):
        self.model_params.update(new_model_params)

    def create_layer(self, layer_type, args=None, kwargs=None):
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        layer_class = self.layer_mapping.get(layer_type)
        if not layer_class:
            raise ValueError(f"Layer type {layer_type} not supported")
        return layer_class(*args, **kwargs)

    def build_from_config(self, component_configs, component_type, input_shape,
                          **kwargs):
        input_layer = Input(shape=(input_shape,),
                            name=f'{component_type}_input')
        x = input_layer
        for i, component_config in enumerate(component_configs):
            layer_type = component_config['type']

            if layer_type == 'bidirectional':
                # Special handling for bidirectional to wrap another layer
                inner_layer_class = self.layer_mapping.get(
                    component_config['inner_type'])
                if not inner_layer_class:
                    raise ValueError(
                        f"Inner layer type {component_config['inner_type']} not supported"
                    )
                inner_layer = inner_layer_class(
                    *component_config.get('args', []),
                    **component_config.get('inner_kwargs', {}))
                layer = Bidirectional(inner_layer,
                                      **component_config.get('kwargs', {}))
            else:
                layer = self.create_layer(layer_type,
                                          component_config.get('args', []),
                                          component_config.get('kwargs', {}))
            if layer_type == 'lstm' and component_type == 'decoder' and i == 1:
                x = layer(x, initial_state=kwargs.get('encoder_states'))
            else:
                x = layer(x)
        return input_layer, x

    def build_model(self):
        enc_inputs, enc_outputs = self.build_from_config(
            self.model_params['layer_config'].get('encoder', []), 'encoder',
            self.model_params['input_shape'])

        encoder_outputs, state_h, state_c = enc_outputs
        encoder_states = [state_h, state_c]  # Assuming LSTM for simplicity
        dec_inputs, dec_outputs = self.build_from_config(
            self.model_params['layer_config'].get('decoder', []),
            'decoder',
            self.model_params['input_shape'],
            encoder_states=encoder_states)
        decoder_outputs, _, _ = dec_outputs
        output_layer_config = self.model_params['layer_config'].get(
            'output_layer', {
                'units': 1,
                'activation': 'softmax',
                'name': 'output_layer'
            })
        output_layer = Dense(**output_layer_config)(decoder_outputs)

        model = Model(inputs=[enc_inputs, dec_inputs],
                      outputs=output_layer,
                      name=self.model_params.get('name', 'seq2seq_model'))

        # Configuring the optimizer
        optimizer_config = self.model_params.get('optimizer', {
            'type': 'adam',
            'learning_rate': 0.001
        })
        optimizer_class = self.layer_mapping.get(optimizer_config['type'])
        if optimizer_class is None:
            optimizer_mapping = {'adam': Adam, 'rmsprop': RMSprop, 'sgd': SGD}
            optimizer_class = optimizer_mapping.get(
                optimizer_config['type'].lower())
        if optimizer_class is None:
            raise ValueError(
                f"Optimizer {optimizer_config['type']} not supported")
        optimizer = optimizer_class(
            **{k: v for k, v in optimizer_config.items() if k != 'type'})
        model.compile(optimizer=optimizer,
                      loss=self.model_params.get(
                          'loss', 'sparse_categorical_crossentropy'),
                      metrics=self.model_params.get('metrics', ['accuracy']))
        return model
