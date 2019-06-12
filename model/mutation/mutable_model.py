from .mutable_base import MutableBase
from model.block import Block

from numpy.random import choice

class MutableModel(MutableBase):
    def __init__(self, raw_dict=None, previous_block = None):

        self.mutation_operators = [("mutate_add_block",0.3),("mutate_block",0.5),("remove_remove_block",0.2)]


    def mutate_add_block(self, block=None):
        block = Block()
        self.blocks.append(block)


    def mutate_block(self, block_index):
        pass

    def mutate_remove_block(self, block=None):
        pass


    def mutate(self):
        e,p =  zip(*self.mutation_operators)
        operation = getattr(self, choice(e, None, p))

        operation()