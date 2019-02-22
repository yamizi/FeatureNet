# -*- coding: utf-8 -*-

from .node import Node

class Operation(Node):
    def __init__(self,  raw_dict=None):
        super(Operation, self).__init__(raw_dict=raw_dict)
        

    def build_tensorflow_model(self, model, source1, source2):
        pass

    @staticmethod
    def parse_feature_model(feature_model):
        pass


class Combination(object):
    def __init__(self, raw_dict=None):
        super(Combination, self).__init__(raw_dict=raw_dict)
        

    def build_tensorflow_model(self, model, source1, source2):
        pass

    @staticmethod
    def parse_feature_model(feature_model):
        pass
