
from numpy.random import choice

class MutableBase(object):
  
    mutation_operators = []
    debug_mode = False
    
    def mutate(self):
        e,p =  zip(*self.mutation_operators)
        operation = getattr(self, choice(e, None, p))

        result = operation()
        if result and MutableBase.debug_mode:
            print("mutation {}".format(result))

    def __init__(self, raw_dict=None, previous_block = None):

        super(MutableBase, self).__init__(raw_dict,previous_block)

    
