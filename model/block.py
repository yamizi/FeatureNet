# -*- coding: utf-8 -*-

from .mutation.mutable_block import MutableBlock
from .node import Node
from .cell import Cell
from .output import Out, OutCell, OutBlock

class Block(MutableBlock, Node):
    def __init__(self, raw_dict=None, previous_block = None):

        self.is_root = True
        self.cells = []
        
        if previous_block:
            self.previous_block = previous_block
            self.is_root = False

        super(Block, self).__init__(raw_dict=raw_dict)

    def append_cell(self, cell):
        self.cells.append(cell)

    def get_custom_parameters(self):
        params = {}
        my_params = self.customizable_parameters
        if len(my_params.keys()):
            params = {self.get_name():(self, my_params)}
        
        for cell in self.cells:
            params = {**params, **cell.get_custom_parameters()}

        return params

    def build_tensorflow_model(self, inputs):
        
        outputs = []
        for cell in self.cells:
            _outputs, _inputs = cell.build_tensorflow_model(inputs)
            
            #Reputting cell inputs that have planned in previous cells 
            for i in _inputs:
                if type(i) is OutCell and i.currentIndex>0:
                    i.currentIndex = i.currentIndex-1
                    if i.currentIndex==0:
                        _inputs.insert(0,i)
            
            if len(outputs)==0:
                outputs = outputs + _outputs
            # We only keep one output
            
        #Cleaning the input stack from the Output who are directed to cells or to be logged out
        _inputs = [i for i in inputs if (i is not Out and i is not OutCell)]
        #Reputting block inputs that have been planned in previous cells 
        for i in _inputs:
            if i is OutBlock:
                i.currentIndex = i.currentIndex-1
                if i.currentIndex==0:
                    _inputs.insert(i)
       
        return outputs, _inputs 


    @staticmethod
    def parse_feature_model(feature_model):
       
        block = Block(raw_dict=feature_model)

        for cell_dict in feature_model.get("children"):
            if len(cell_dict.get("children")):
                cell_type = cell_dict.get("children")[0].get("label")
                cell_type = cell_type[cell_type.rfind("_")+1:]
                cell_type = ''.join([i for i in cell_type if not i.isdigit()])
                
                if cell_type=="Cell":
                    cell = Cell.parse_feature_model(cell_dict.get("children")[0])
                    block.cells.append(cell)
        
        block.cells.sort(key = lambda a : a.get_name())
        return block
