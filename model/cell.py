# -*- coding: utf-8 -*-

from .node import Node
from .input import Input
from .output import Output
from .operation import Operation, Combination

class Cell(Node):
    def __init__(self, raw_dict=None, input1=None, operation1=None, input2=None, operation2=None, output=None, output_combination=None):

        self.input1 = input1
        self.input2 = input2
        self.operation1 = operation1
        self.operation2 = operation2
        self.output = output
        self.combination = output_combination
        super(Cell, self).__init__(raw_dict=raw_dict)

    def build_tensorflow_model(self, model, source1, source2):
        pass

    def get_custom_parameters(self):
        my_params = self.customizable_parameters
        params = {}
        if len(my_params.keys()):
            params = {self.get_name():(self, my_params)}

        params = {**params, **self.input1.get_custom_parameters(), **self.input2.get_custom_parameters() , **self.operation1.get_custom_parameters(), **self.operation2.get_custom_parameters() , **self.combination.get_custom_parameters()}
        return params

    @staticmethod
    def parse_feature_model(feature_model):
        cell = Cell(raw_dict=feature_model)

        for cell_element_dict in feature_model.get("children"):
            element_type = Node.get_type(cell_element_dict)
            element = None
            
            if(element_type=="input1"):
                element = Input.parse_feature_model(cell_element_dict)

            elif(element_type=="input2"):
                element = Input.parse_feature_model(cell_element_dict)
                
            elif(element_type=="operation1"):
                element = Operation.parse_feature_model(cell_element_dict)
                
            elif(element_type=="operation2"):
                element = Operation.parse_feature_model(cell_element_dict)
                
            elif(element_type=="combination"):
                element = Combination.parse_feature_model(cell_element_dict)
                
            elif(element_type=="output"):
                element = Output.parse_feature_model(cell_element_dict)
                
            setattr(cell, element_type, element)   
            print("settings {0}".format(element_type)) 
  

        return cell
        