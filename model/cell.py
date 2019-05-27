# -*- coding: utf-8 -*-
from .node import Node
from .input import Input, ZerosInput, IdentityInput
from .output import Output, OutCell, OutBlock, Out
from .operation import Operation, Combination, Sum, Void

class Cell(Node):
    def __init__(self, raw_dict=None, input1=None, operation1=None, input2=None, operation2=None, output=None, output_combination=None):

        if not input1:
            input1 = IdentityInput()
        if not input2:
            input2 = ZerosInput()
        if not operation1:
            operation1 = Void()
        if not operation2:
            operation2 = Void()
        if not output_combination:
            output_combination = Sum()
        if not output:
            output = OutCell()

        self.input1 = input1
        self.input2 = input2
        self.operation1 = operation1
        self.operation2 = operation2
        self.output = output
        self.combination = output_combination
        super(Cell, self).__init__(raw_dict=raw_dict)

    def build_tensorflow_model(self, inputs, max_relative_index):

        last_inputs = [input for input in inputs if not hasattr(input,"content") or input.currentIndex ==0]

        i1 = self.input1.build(last_inputs[0] if len(last_inputs)>0 else None)
        i1 = self.operation1.build(i1)
       
        if type(self.input2) is ZerosInput:
            combination = i1
        else:
            i2 = self.input2.build(last_inputs[1] if len(last_inputs)>1 else last_inputs[0] if len(last_inputs)>0 else None, i1)
            i2 = self.operation2.build(i2)
            combination = self.combination.build(i1,i2)

        self.output.currentIndex = min(max_relative_index,self.output.currentIndex)
        output = self.output.build(combination)
        outputs = []
        if type(self.output) is OutCell and self.output.currentIndex>-1:           
            inputs.insert(0, output)
        if type(self.output) is Out:
            outputs.insert(0, output.content)

        return outputs

    def get_custom_parameters(self):
        my_params = self.customizable_parameters
        params = {}
        if len(my_params.keys()):
            params = {self.get_name():(self, my_params)}

        params = {**params, **self.input1.get_custom_parameters(), **self.input2.get_custom_parameters() , **self.operation1.get_custom_parameters(), **self.operation2.get_custom_parameters() , **self.combination.get_custom_parameters()}
        return params

    @staticmethod
    def parse_feature_model(feature_model):
        feature_model["children"] = list(reversed(feature_model["children"]))
        cell = Cell(raw_dict=feature_model)

        for cell_element_dict in feature_model.get("children"):
            element_type = Node.get_type(cell_element_dict)
            element = None

            if len(cell_element_dict.get("children")) >0:
                
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
            
            if element:
                setattr(cell, element_type, element)   
            #print("settings {0} {1}".format(element_type, element.get_name())) 
  
        return cell
        