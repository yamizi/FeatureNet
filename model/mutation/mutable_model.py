from .mutable_base import MutableBase, MutationStrategies
from ..block import Block

from numpy.random import choice, rand

class MutableModel(MutableBase):
    def __init__(self):
        self.mutation_operators = [("mutate_add_block",0.3),("mutate_block",0.5),("mutate_remove_block",0.2)]

    def mutate_add_block(self, rate=1, block=None):
        prob = rand()
        if MutableBase.mutation_stategy==MutationStrategies.CHOICE or prob < rate:
            block = Block.base_block()
            self.blocks.append(block)

            return ("add_block",block)

    def mutate_block(self, rate=1, block_index=None):
        #print("index {}".format(block_index))
        if len(self.blocks) ==0:
            return ("mutate_block",None)

        if block_index is None:
            if MutableBase.mutation_stategy==MutationStrategies.CHOICE:
                block_index  = choice(len(self.blocks))
                return self.mutate_block(rate, block_index)
            else:
                for block_index, block in enumerate(self.blocks):
                    self.mutate_block(rate, block_index)
        else:
            block = self.blocks[block_index]
            return block.mutate()

    def mutate_remove_block(self, rate=1, block_index=None):
        if len(self.blocks) >=1:
            return ("remove_block",None)
        if block_index is None:
            if MutableBase.mutation_stategy==MutationStrategies.CHOICE:
                block_index  = choice(len(self.blocks))
                return self.mutate_remove_block(rate,block_index)
            else:
                for block_index, block in enumerate(self.blocks):
                    prob = rand()
                    if prob < rate:
                        self.mutate_remove_block(rate, block_index)

        elif block_index >=0 and block_index<len(self.blocks):
            del self.blocks[block_index]
            return ("remove_block",block_index)