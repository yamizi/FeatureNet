# -*- coding: utf-8 -*-
""""""
from __future__ import absolute_import, division, print_function, unicode_literals

import keras.backend as k
from keras.models import Sequential, Model
from keras.layers import Input
#from keras import optimizers

from .block import Block

class KerasFeatureModel(object):
    
    blocks = []
    outputs = []
    optimizers = []
    losss = ['categorical_crossentropy']


    def __init__(self, name=""):
        self._name = name

    def get_custom_parameters(self):
        params = {}
               
        for block in self.blocks:
            params = {**params, **block.get_custom_parameters()}

        return params

        
    def build(self, input_shape):
        self.outputs = []

        X_input = Input(input_shape)
        _inputs = [X_input]

        for block in self.blocks:
            _outputs, _inputs = block.build_tensorflow_model(_inputs)
            self.outputs = self.outputs + _outputs


        # Create model
        model = Model(outputs=self.outputs, inputs=X_input,name=self._name)

        #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=self.losss[0], metrics=['accuracy'], optimizer=self.optimizers[0] if len(self.optimizers) else "sgd")

        return model


    @staticmethod
    def parse_feature_model(feature_model, name=None):

        print("building keras model from feature model tree")
        model = KerasFeatureModel(name=name)

        for block_dict in feature_model:
            block = Block.parse_feature_model(block_dict)
            model.blocks.append(block)

        model.blocks.sort(key = lambda a : a.get_name())

        missing_params = model.get_custom_parameters()
        for name,(node, params) in missing_params.items():
            print("{0}:{1}".format(name, params))

        return model
            