
from numpy.random import choice

class MutableBase(object):
  
    mutation_operators = []
    
    def mutate(self):
        e,p =  zip(*self.mutation_operators)
        operation = getattr(self, choice(e, None, p))

        operation()

    def __init__(self, raw_dict=None, previous_block = None):

        super(MutableBase, self).__init__(raw_dict,previous_block)

    
