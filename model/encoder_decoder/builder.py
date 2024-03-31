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
        embedding_layer = [
            component_config for component_config in component_configs
            if component_config['type'] == 'embedding'
        ][0]
        layer = self.create_layer(embedding_layer['type'],
                                  args=embedding_layer['args'],
                                  kwargs=embedding_layer['kwargs'])
        x = layer(input_layer)
        x = (x,)
        component_configs = [
            component_config for component_config in component_configs
            if component_config['type'] != 'embedding'
        ]
        for i, component_config in enumerate(component_configs):
            layer_type = component_config['type']
            layer = self.create_layer(layer_type,
                                      component_config.get('args', []),
                                      component_config.get('kwargs', {}))
            if layer_type == 'lstm' and component_type == 'decoder' and i == 0:
                x = layer(x[0], initial_state=kwargs.get('encoder_states'))
            else:
                x = layer(x[0])
        return input_layer, x

    def configure_optimizer(self, optimizer_config):
        optimizer_mapping = {'adam': Adam, 'rmsprop': RMSprop, 'sgd': SGD}
        optimizer_class = optimizer_mapping.get(
            optimizer_config['type'].lower())
        if optimizer_class is None:
            raise ValueError(
                f"Optimizer {optimizer_config['type']} not supported")
        optimizer = optimizer_class(
            **{k: v for k, v in optimizer_config.items() if k != 'type'})
        return optimizer

    def build_model(self):
        print("hello")
        enc_inputs, enc_outputs = self.build_from_config(
            self.model_params['layer_config'].get('encoder', []), 'encoder',
            self.model_params['encoder_input_len'])
        encoder_outputs, state_h, state_c = enc_outputs
        encoder_states = [state_h, state_c]  # Assuming LSTM for simplicity
        encoder_model = Model(inputs=enc_inputs,
                              outputs=encoder_states,
                              name='encoder_model')

        dec_inputs, dec_outputs = self.build_from_config(
            self.model_params['layer_config'].get('decoder', []),
            'decoder',
            self.model_params['decoder_input_len'],
            encoder_states=encoder_states)
        decoder_outputs, _, _ = dec_outputs
        output_layer_config = self.model_params['layer_config'].get(
            'output_layer', {
                'units': 1,
                'activation': 'softmax',
                'name': 'output_layer'
            })
        output_layer = Dense(**output_layer_config)(decoder_outputs)
        decoder_model = Model(inputs=[dec_inputs] + encoder_states,
                              outputs=output_layer,
                              name='decoder_model')

        model = Model(inputs=[enc_inputs, dec_inputs],
                      outputs=output_layer,
                      name=self.model_params.get('name', 'seq2seq_model'))

        # Configuring the optimizer
        optimizer_config = self.model_params.get('optimizer', {
            'type': 'adam',
            'learning_rate': 0.001
        })
        optimizer = self.configure_optimizer(optimizer_config)
        model.compile(optimizer=optimizer,
                      loss=self.model_params.get(
                          'loss', 'sparse_categorical_crossentropy'),
                      metrics=self.model_params.get('metrics', ['accuracy']))
        return model, encoder_model, decoder_model
