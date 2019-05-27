# -*- coding: utf-8 -*-

from .node import Node
from .cell import Cell
from .output import Out, OutCell, OutBlock

class Block(Node):
    def __init__(self, raw_dict=None, previous_block = None):

        self.is_root = True
        self.cells = []
        self._features = 0.0
        self._stride = 0
        
        if previous_block:
            self.previous_block = previous_block
            self.is_root = False

        super(Block, self).__init__(raw_dict=raw_dict)

    def append_cell(self, cell):
        self.cells.append(cell)


    def set_stride(self,stride_str):
        self._stride = stride_str.split("x")

    def set_features(self,features_str):
        self._features = int(features_str)/100

    def get_custom_parameters(self):
        params = {}
        my_params = self.customizable_parameters
        if len(my_params.keys()):
            params = {self.get_name():(self, my_params)}
        
        for cell in self.cells:
            params = {**params, **cell.get_custom_parameters()}

        return params

    def build_tensorflow_model(self, inputs):
        
        block_input = inputs[0]
        outputs = []
        nb_cells =len(self.cells)
        for i, cell in enumerate(self.cells):
            max_relative_index = nb_cells -i
            _outputs = cell.build_tensorflow_model(inputs,max_relative_index)
            
            #Reputting cell inputs that have planned in previous cells 
            for i in inputs:
                if type(i) is OutCell and i.currentIndex>=0:
                    i.currentIndex = i.currentIndex-1
            
            outputs = outputs + _outputs
            # To handle multiple outputs if the feature model includes multiples
            
        #Cleaning the input stack from the Output who are directed to cells or to be logged out => Keeping only BlockOutput
        #If no Block Output defined, building a new one based on the last cell
        _inputs = [i for i in inputs if i is OutBlock]
        if len(_inputs) ==0:
            outBlock = OutBlock()
            outBlock.parent_name = self.get_name()

            outBlock.build(inputs[0])
            _inputs = [outBlock]
        
        _inputs.append(block_input)

        return _inputs, outputs 


    @staticmethod
    def parse_feature_model(feature_model):
        
        block = Block(raw_dict=feature_model)
        
        for cell_dict in feature_model.get("children"):
            element_type = cell_dict.get("label")
            element_type = element_type[element_type.rfind("_")+1:]
            if element_type=="stride":
                stride = cell_dict.get("children")[0].get("label")
                block.set_stride(stride[stride.rfind("_")+1:])
            elif element_type=="features":
                feature = cell_dict.get("children")[0].get("label")
                block.set_features(feature[feature.rfind("_")+1:])

            elif len(cell_dict.get("children")):

                
                        cell_type = cell_dict.get("children")[0].get("label")
                        cell_type = cell_type[cell_type.rfind("_")+1:]
                        cell_type = ''.join([i for i in cell_type if not i.isdigit()])
                
                        if cell_type=="Cell":
                            cell = Cell.parse_feature_model(cell_dict.get("children")[0])
                            block.cells.append(cell)
        
        block.cells.sort(key = lambda a : a.get_name())
        return block
