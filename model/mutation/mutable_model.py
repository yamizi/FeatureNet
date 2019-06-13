from .mutable_base import MutableBase
from model.block import Block
from model.cell import Cell

from numpy.random import choice

class MutableModel(MutableBase):
    def __init__(self, raw_dict=None, previous_block = None):
        self.mutation_operators = [("mutate_add_block",0.3),("mutate_block",0.5),("remove_remove_block",0.2)]

    def mutate_add_block(self, block=None):
        block = Block.base_block()
        self.blocks.append(block)

        return ("add_block",block)

    def mutate_block(self, block_index):
        block = self.blocks[block_index]
        block.mutate()

    def mutate_remove_block(self, block_index=None):
        if block_index >0 and block_index<len(self.blocks):
            del self.blocks[block_index]
            return ("remove_block",block_index)