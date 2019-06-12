from .mutable_base import MutableBase

class MutableBlock(MutableBase):
    def __init__(self, raw_dict=None, previous_block = None):


        self.mutation_operators = {
            

        }
        super(MutableBlock, self).__init__(raw_dict,previous_block)


    def mutate_add_block(self, block=None):
        pass

    def mutate_remove_block(self, block=None):
        pass


    def mutate(self, mutation_rate=0.5):
