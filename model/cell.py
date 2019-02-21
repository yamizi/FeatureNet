# -*- coding: utf-8 -*-


class Cell(object):
    def __init__(self, input1, operation1, input2, operation2, output, output_combination):

        self.input1 = input1
        self.input2 = input2
        self.operation1 = operation1
        self.operation2 = operation2
        self.output = output
        self.output_combination = output_combination
        

    def build_tensorflow_model(self, model, source1, source2):
        pass

    @staticmethod
    def parse_feature_mode(feature_model):
        print(feature_model)