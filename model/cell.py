# -*- coding: utf-8 -*-

from .node import Node
from .input import Input, ZerosInput
from .output import Output, OutCell, OutBlock, Out
from .operation import Operation, Combination, Sum

class Cell(Node):
    def __init__(self, raw_dict=None, input1=None, operation1=None, input2=None, operation2=None, output=None, output_combination=None):

        self.input1 = input1
        self.input2 = input2
        self.operation1 = operation1
        self.operation2 = operation2
        self.output = output
        self.combination = output_combination
        super(Cell, self).__init__(raw_dict=raw_dict)

    def build_tensorflow_model(self, inputs):

        last_inputs = inputs #[input for input in inputs if  input is not Output or input.currentIndex ==1]

        i1 = self.input1.build(last_inputs[0] if len(last_inputs)>0 else None)
        i1 = self.operation1.build(i1)
       
        if type(self.input2) is ZerosInput:
            combination = i1
        else:
            i2 = self.input2.build(last_inputs[1] if len(last_inputs)>1 else last_inputs[0] if len(last_inputs)>0 else None, i1)
            i2 = self.operation1.build(i2)
            combination = self.combination.build(i1,i2)

        output = self.output.build(combination)
        outputs = []
        if type(self.output) is OutCell and self.output.currentIndex==1:           
            inputs.insert(0, output)
        if type(self.output) is Out:
            outputs = [output.content]

        return outputs, inputs

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
        