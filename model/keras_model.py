# -*- coding: utf-8 -*-
""""""
from __future__ import absolute_import, division, print_function, unicode_literals

import keras.backend as k
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten
from keras.layers import Input
from keras.optimizers import SGD
#from keras import optimizers

from .block import Block
from .output import Out
from .cell import Cell

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

        
    def build(self, input_shape, output_shape):
        self.outputs = []

        X_input = Input(input_shape)
        _inputs = [X_input]
        model = None

        self.optimizers.append(SGD(lr=0.1, momentum=0.9, decay=0.0001, nesterov=True))
        self.optimizers = ["sgd"]
        
        try:
            print("Build Tensorflow model")
            for block in self.blocks:
                _outputs, _inputs = block.build_tensorflow_model(_inputs)
                self.outputs = self.outputs + _outputs

            out = self.outputs[-1]
            out = out.content if hasattr(out,"content") else out

            if out.shape.ndims >2:
                out = Flatten()(out)
            self.outputs = [Dense(output_shape, activation="softmax", name="out")(out)]
            # Create model
            model = Model(outputs=self.outputs, inputs=X_input,name=self._name)

            #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss=self.losss[0], metrics=['accuracy'], optimizer=self.optimizers[0] if len(self.optimizers) else "sgd")
        
        except Exception as e:
            print(e)
        return model


    @staticmethod
    def parse_feature_model(feature_model, name=None, depth=1):

        print("building keras model from feature model tree")
        model = KerasFeatureModel(name=name)
        model.blocks = []

        if len(feature_model)==0:
            return model

        if isinstance(feature_model, str):
            model.blocks = KerasFeatureModel.get_from_template(feature_model)

        else: 
            for i in range(depth):
                for block_dict in feature_model:
                    block = Block.parse_feature_model(block_dict)
                    model.blocks.append(block)

            model.blocks.sort(key = lambda a : a.get_name())

            missing_params = model.get_custom_parameters()
            for name,(node, params) in missing_params.items():
                print("{0}:{1}".format(name, params))

        return model


    @staticmethod
    def get_from_template(feature_model):
        blocks = []
        if feature_model=="lenet5":
            blocks =  KerasFeatureModel.lenet5_blocks()
        
        return blocks

    @staticmethod
    def lenet5_blocks():
        blocks = []

        from .input import Input, ZerosInput, PoolingInput, ConvolutionInput, DenseInput, IdentityInput
        from .output import Output, OutCell, OutBlock, Out
        from .operation import Operation, Combination, Sum, Flat

        block1 = Block()
        cell11 = Cell(input1 = ConvolutionInput((5,5),(1,1),6,"same", "tanh"))
        block1.append_cell(cell11)
        cell12 = Cell(input1 = PoolingInput((2,2),(1,1),"average", "valid"), output=OutBlock())
        block1.append_cell(cell12)

        block2 = Block()
        cell21 = Cell(input1 = ConvolutionInput((5,5),(1,1),6,"same", "tanh"))
        block2.append_cell(cell21)
        cell22 = Cell(input1 = PoolingInput((2,2),(2,2),"average", "valid"), output=OutBlock())
        block2.append_cell(cell22)


        block3 = Block()
        cell31 = Cell(input1 = ConvolutionInput((5,5),(1,1),120,"valid", "tanh"), operation1=Flat(), output=OutBlock())
        block3.append_cell(cell31)

        block4 = Block()
        cell41 = Cell(input1 = DenseInput(84, "tanh"), output=Out())
        block4.append_cell(cell41)

        blocks.extend([block1, block2, block3, block4])

        return blocks

    
