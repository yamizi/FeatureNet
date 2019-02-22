# -*- coding: utf-8 -*-

from .node import Node

class Output(object):
    def __init__(self, raw_dict=None):
        super(Output, self).__init__(raw_dict=raw_dict)
        

    def build_tensorflow_model(self, model, source1, source2):
        pass

    @staticmethod
    def parse_feature_model(feature_model):
        pass

