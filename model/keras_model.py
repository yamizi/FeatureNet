# -*- coding: utf-8 -*-
""""""
from __future__ import absolute_import, division, print_function, unicode_literals

import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

from .block import Block

class KerasFeatureModel(Sequential):
    
    blocks = []

    def __init__(self, layers=None, name=None):
        super(KerasFeatureModel, self).__init__(layers=None, name=None)

    def get_custom_parameters(self):
        params = {}
               
        for block in self.blocks:
            params = {**params, **block.get_custom_parameters()}

        return params

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
            print("{0}:{1}".format(name, list(params.keys())))
            