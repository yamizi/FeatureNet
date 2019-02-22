# -*- coding: utf-8 -*-

from .node import Node
from .cell import Cell

class Block(Node):
    def __init__(self, raw_dict=None, previous_block = None):

        self.is_root = True
        self.cells = []
        
        if previous_block:
            self.previous_block = previous_block
            self.is_root = False

        super(Block, self).__init__(raw_dict=raw_dict)

    def append_cell(self, cell):
        self.cells.append(cell)


    def build_tensorflow_model(self, model):
        pass


    @staticmethod
    def parse_feature_model(feature_model):
       
        block = Block(raw_dict=feature_model)

        for cell_dict in feature_model.get("children"):
            cell_type = cell_dict.get("children")[0].get("label")
            cell_type = cell_type[cell_type.rfind("_")+1:]
            cell_type = ''.join([i for i in cell_type if not i.isdigit()])
            
            if cell_type=="Cell":
                cell = Cell.parse_feature_model(cell_dict.get("children")[0])
                block.cells.append(cell)
        
        block.cells.sort(key = lambda a : a.get_name())
        return block
