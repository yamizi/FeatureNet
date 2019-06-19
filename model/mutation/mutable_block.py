from .mutable_base import MutableBase, MutationStrategies
from ..cell import Cell
from numpy.random import choice, rand

class MutableBlock(MutableBase):
    cells = []
    attributes = {"strides_values":None,"features_multiplier_values":None}
    strides_values = ("1x1","2x2")
    features_multiplier_values = (800,400,200,100,50,25)

    def __init__(self, raw_dict=None, previous_block = None, parent_model=None):

        self.mutation_operators = (("mutate_add_cell",0.3),("mutate_cell",0.3),("mutate_remove_cell",0.1),("mutate_block",0.3))
        super(MutableBlock, self).__init__(raw_dict,previous_block)


    def mutate_add_cell(self, rate=1):
        
        prob = rand()
        if prob < rate or MutableBase.mutation_stategy==MutationStrategies.CHOICE:
            cell = Cell.base_cell()
            self.cells.append(cell)

        return ("mutate_add_cell",cell)

    def mutate_block(self, rate=1):
        choices = list(self.attributes.keys())
        returns = []
        if MutableBase.mutation_stategy==MutationStrategies.CHOICE:
            attribute_to_mutate = choice(choices, None)
            attribute_value = choice(getattr(self,attribute_to_mutate), None)
            self.attributes[attribute_to_mutate](attribute_value)

            returns.append(("mutate_block",attribute_to_mutate,attribute_value ))
        else:
            for attribute_to_mutate in choices:
                prob = rand()
                if prob < rate:
                    attribute_value = choice(getattr(self,attribute_to_mutate), None)
                    self.attributes[attribute_to_mutate](attribute_value)
                    returns.append(("mutate_block",attribute_to_mutate,attribute_value ))

        return returns

    def mutate_cell(self, rate=1, cell_index=None):
        if len(self.cells) ==0:
            return ("mutate_cell",None)

        if cell_index is not None:
            cell = self.cells[cell_index]
            cell.mutate(rate)
        elif MutableBase.mutation_stategy==MutationStrategies.CHOICE:
            cell_index  = choice(len(self.cells))
            return self.mutate_cell(rate, cell_index)
        else:
            for cell_index, cell in enumerate(self.cells):
                self.mutate_cell(rate, cell_index)

    def mutate_remove_cell(self,rate=1, cell_index=None):
        if len(self.cells) ==0:
            return ("mutate_remove_cell",None)
        if cell_index is not None and cell_index >=0 and cell_index<len(self.cells):
            del self.cells[cell_index]
            return ("mutate_remove_cell",cell_index)

        elif MutableBase.mutation_stategy==MutationStrategies.CHOICE:
            cell_index  = choice(len(self.cells))
            return self.mutate_remove_cell(rate, cell_index)
        else:
            for cell_index, cell in enumerate(self.cells):
                prob = rand()
                if prob < rate:
                    return self.mutate_remove_cell(rate, cell_index)

       