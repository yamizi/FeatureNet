from .mutable_base import MutableBase
from model.cell import Cell
from numpy.random import choice

class MutableBlock(MutableBase):
    cells = []
    attributes = {"strides_values":None,"features_multiplier_values":None}
    strides_values = ("1x1","2x2")
    features_multiplier_values = (800,400,200,100,50,25)

    def __init__(self, raw_dict=None, previous_block = None):

        self.mutation_operators = (("mutate_add_cell",0.2),("mutate_cell",0.4),("mutate_block",0.3),("remove_remove_cell",0.1))
        super(MutableBlock, self).__init__(raw_dict,previous_block)


    def mutate_add_block(self, block=None):
        cell = Cell.base_cell()
        self.cells.append(cell)

        return ("mutate_add_block",cell)

    def mutate_block(self):
        attribute_to_mutate = getattr(self,choice(self.attributes.keys(), None))
        attribute_value = choice(attribute_to_mutate, None)
        self.attributes[attribute_to_mutate](attribute_value)

        return ("mutate_block",attribute_to_mutate,attribute_value )

    def mutate_cell(self, cell_index):
        cell = self.cells[cell_index]
        cell.mutate()

    def mutate_remove_block(self, cell_index=None):
        if cell_index >0 and cell_index<len(self.cells):
            del self.cells[cell_index]
            return ("mutate_remove_block",cell_index)