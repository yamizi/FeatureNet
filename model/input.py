# -*- coding: utf-8 -*-

from .node import Node

class Input(Node):
    def __init__(self, raw_dict=None):
        
        self.raw_dict = raw_dict
        super(Input, self).__init__(raw_dict=raw_dict)

    def build_tensorflow_model(self, model, source1, source2):
        pass

    @staticmethod
    def parse_feature_model(feature_model):

        input = feature_model.get("children")[0] 
        input_type = Node.get_type(input)

        if(input_type=="dense"):
            print(input.get("children"))
            #activation = 
            #input_element = DenseInput(raw_dict=feature_model)
        
        return input

class ZerosInput(Input):
    def __init__(self, raw_dict=None):
        super(ZerosInput, self).__init__(raw_dict=raw_dict)

class IdentityInput(Input):
    def __init__(self, raw_dict=None):
        super(IdentityInput, self).__init__(raw_dict=raw_dict)

class DenseInput(Input):
    def __init__(self, features, activation, raw_dict=None):
        super(DenseInput, self).__init__(raw_dict=raw_dict)

class PoolingInput(Input):
    def __init__(self, features, stride, type, padding, raw_dict=None):
        super(PoolingInput, self).__init__(raw_dict=raw_dict)

class ConvolutionInput(Input):
    def __init__(self, kernel, stride, features, padding, activation, raw_dict=None):
        super(ConvolutionInput, self).__init__(raw_dict=raw_dict)