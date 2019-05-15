# -*- coding: utf-8 -*-
""""""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import keras.backend as k
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten
from keras.layers import Input, GlobalAveragePooling2D
from keras.optimizers import SGD
#from keras import optimizers

from .block import Block
from .output import Out
from .cell import Cell

import random


class KerasFeatureVector(object):

    features = []
    attributes = [0,0,0,0]
    accuracy = None

    def __init__(self, accuracy, attributes, features):
        self.features = features
        self.attributes = attributes
        self.accuracy= accuracy

    def mutate(self, rate=0.05):
        l = len(self.features)
        mask = [random.random() > rate for _ in range(l)]
        self.features = [self.features[i]  if mask[i] else 1 - self.features[i] for i in range(l)]   
 
    def cross_over(self, second_vector, crossover_type="onepoint"):
        if crossover_type=="onepoint":
            point = random.randint(0, len(self.features))
            return KerasFeatureVector(0, [0,0, 0, 0], self.features[0:point]+second_vector.features[point:])

    def to_vector(self):
        return [self.accuracy]+ self.attributes + self.features

    @staticmethod
    def from_vector(vect):
        return KerasFeatureVector(vect[0], [vect[1],vect[2], vect[3], vect[4]], vect[5:])

    def __str__(self):
        return "{}:{}".format(";".join([str(i) for i in self.attributes]), self.accuracy)

    @property
    def fitness(self):
        return 0 if self.accuracy is None else self.accuracy 

class KerasFeatureModel(object):
    
    blocks = []
    outputs = []
    optimizers = []
    features = []
    features_label=  []
    nb_flops  = 0
    nb_params = 0
    model = None
    accuracy = 0

    losss = ['categorical_crossentropy']


    def __init__(self, name=""):
        self._name = name

    def get_custom_parameters(self):
        params = {}
               
        for block in self.blocks:
            params = {**params, **block.get_custom_parameters()}

        return params

    
    def to_vector(self):
        if self.model:
            nb_layers = len(self.model.layers)
        else:
            nb_layers = 0
            
        return KerasFeatureVector(self.accuracy, [len(self.blocks),nb_layers, self.nb_params, self.nb_flops], self.features)

        
    def build(self, input_shape, output_shape, max_parameters=20000000):
        self.outputs = []

        X_input = Input(input_shape)
        _inputs = [X_input]
        model = None

        lr=1.e-2
        n_steps=20
        global_step = tf.Variable(0)    
        global_step=1
        learning_rate = tf.train.cosine_decay(
            learning_rate=lr,
            global_step=global_step,
            decay_steps=n_steps
        )
        self.optimizers.append(SGD(lr=0.1, momentum=0.9, decay=0.0001, nesterov=True))
        self.optimizers = [ "sgd", tf.train.RMSPropOptimizer(learning_rate=learning_rate)]
        
        try:
            print("Build Tensorflow model")
            for block in self.blocks:
                _outputs, _inputs = block.build_tensorflow_model(_inputs)
                self.outputs = self.outputs + _outputs

            out = self.outputs[-1] if len(self.outputs) else  _inputs[0]
            out = out.content if hasattr(out,"content") else out

            if out.shape.ndims >2:
                #out = Flatten()(out)
                out = GlobalAveragePooling2D()(out)
            self.outputs = [Dense(output_shape, activation="softmax", name="out")(out)]
            # Create model
            model = Model(outputs=self.outputs, inputs=X_input,name=self._name)

            #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

            if model.count_params() > 20000000:
                print("#### model is bigger than 20M params. Skipped")
                model.summary()
                return None 

            model.compile(loss=self.losss[0], metrics=['accuracy'], optimizer=self.optimizers[0] if len(self.optimizers) else "sgd")
        
        except Exception as e:
            print("error",e)
            model.summary()
            return None
        
        self.model = model
        return model


    @staticmethod
    def parse_feature_model(feature_model, name=None, depth=1, product_features=None, features_label=None):

        print("building keras model from feature model tree")
        model = KerasFeatureModel(name=name)

        if product_features:
            model.features = [1 if str(x).isdigit() and int(x)>0 else 0 for x in product_features]
        
        if features_label:
            model.features_label = features_label

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
        cell21 = Cell(input1 = ConvolutionInput((5,5),(1,1),16,"same", "tanh"))
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

    
