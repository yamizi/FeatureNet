# -*- coding: utf-8 -*-
""""""
from __future__ import absolute_import, division, print_function, unicode_literals
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten
from keras.layers import Input, GlobalAveragePooling2D

from .mutation.mutable_model import MutableModel

from keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib
from .block import Block

import random
import uuid

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
        return [self.accuracy]+ [self.attributes] + self.features

    @staticmethod
    def from_vector(vect):
        return KerasFeatureVector(vect[0], vect[1], vect[2:])

    def __str__(self):
        return "{}:{}".format(";".join([str(i) for i in self.attributes]), self.accuracy)

    @property
    def fitness(self):
        return 0 if self.accuracy is None else self.accuracy 

class KerasFeatureModel(MutableModel):
    
    blocks = []
    outputs = []
    optimizers = []
    features = []
    features_label=  []
    nb_flops  = 0
    nb_params = 0
    robustness_score = 0
    clever_score = 0
    pgd_score = 0
    cw_score = 0
    fgsm_score = 0
    model = None
    accuracy = 0
    use_multigpu = True
    robustness_scores = ["clever","cw", "fgsm", "pgd"]
    metrics = []
    
    layers = {"pool":[],"conv":[]}
    


    def __init__(self, name=""):
        if not name:
            name = str(uuid.uuid1())[:10]
        self._name = name
        self.mutation_history = []
        super(KerasFeatureModel, self).__init__()

    def get_custom_parameters(self):
        params = {}
               
        for block in self.blocks:
            params = {**params, **block.get_custom_parameters()}

        return params

    
    def to_kerasvector(self):
        if self.model:
            nb_layers = len(self.model.layers)
        else:
            nb_layers = 0
            
        return KerasFeatureVector(self.accuracy, [self._name, [len(self.blocks),nb_layers, self.nb_params,self.nb_flops], [self.robustness_score,  [self.clever_score, self.fgsm_score, self.pgd_score, self.cw_score]],self.metrics], self.features)
        
    def build(self, input_shape, output_shape, max_parameters=20000000, output_activation="softmax"):
        self.outputs = []

        X_input = Input(input_shape)
        _inputs = [X_input]
        model = None
        
        try:
            print("Build Tensorflow model")
            for block in self.blocks:
                _inputs, _outputs = block.build_tensorflow_model(_inputs)
                self.outputs = self.outputs + _outputs

            out = self.outputs[-1] if len(self.outputs) else  _inputs[0]
            out = out.content if hasattr(out,"content") else out

            if out.shape.ndims >2:
                out = Flatten()(out)
                #out = GlobalAveragePooling2D()(out)
            self.outputs = [Dense(output_shape, activation=output_activation, name="out")(out)]

            model = Model(outputs=self.outputs, inputs=X_input,name=self._name)
            if model.count_params() > 20000000:
                print("#### model is bigger than 20M params. Skipped")
                model.summary()
                return None 

            # if model.count_params() < 20000:
            #     print("#### model is smaller than 20K params. Skipped")
            #     model.summary()
            #     return None 

            if KerasFeatureModel.use_multigpu:
                try:
                    local_device_protos = device_lib.list_local_devices()
                    gpu_devices = [device for device in local_device_protos if device.device_type=="GPU"]
                    if len(gpu_devices) <2:
                        raise Exception()
                    model = multi_gpu_model(model, gpus=len(gpu_devices))
                except Exception as e:
                    print("multi gpu not available")
                    KerasFeatureModel.use_multigpu = False

            
        except Exception as e:
            import traceback
            print("error",e)
            print (traceback.format_exc())
            
            if model:
                model.summary()
            return None
        
        self.model = model
        return model


    def dump_blocks(self):
        return {"blocks":self.blocks, "accuracy":self.accuracy}

    @staticmethod
    def parse_blocks(block_model, name=None):
        #print("building keras model from block list")
        model = KerasFeatureModel(name=name)

        model.blocks = block_model.get("blocks", [])
        model.accuracy = block_model.get("accuracy", 0)

        return model

    @staticmethod
    def parse_feature_model(feature_model, name=None, depth=1, product_features=None, features_label=None):

        print("building keras model from feature model tree {}".format(name))
        model = KerasFeatureModel(name=name)

        if product_features:
            #sorted_features = sorted( product_features, key=lambda k: abs(int(k)))
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
                    block_dict["children"] = list(reversed(block_dict["children"]))
                    block = Block.parse_feature_model(block_dict, model)
                    model.blocks.append(block)

            model.blocks.sort(key = lambda a : a.get_name())

        return model


    @staticmethod
    def get_from_template(feature_model):
        blocks = []
        if feature_model=="lenet5":
            from .leNet import lenet5_blocks
            blocks =  lenet5_blocks()

        if feature_model=="keras":
            from .kerasNet import standard_blocks
            blocks =  standard_blocks()

        if feature_model=="keras_c1d":
            from .kerasNet import cnn1d_blocks
            blocks =  cnn1d_blocks()

        if feature_model=="keras_lstm":
            from .kerasNet import lstm_blocks
            blocks =  lstm_blocks()
        
        return blocks
