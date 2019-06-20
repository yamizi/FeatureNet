from enum import Enum
from numpy.random import choice

class MutationStrategies(Enum):
    CHOICE = 1 # at every node choose only one of the children to mutate
    ALL = 2 # at every node mutate all the children according to mutation rate

class MutableBase(object):
  
    mutation_operators = []
    debug_mode = False
    mutation_stategy = MutationStrategies.CHOICE
    MAX_NB_CELLS = 10
    MAX_NB_BLOCKS = 20
    
    def mutate(self, rate=1):
        e,p =  zip(*self.mutation_operators)
        if MutableBase.mutation_stategy==MutationStrategies.CHOICE:
            operation = getattr(self, choice(e, None, p))

            result = operation(rate)
            if result and MutableBase.debug_mode:
                print("mutation {}".format(result))
        else:
            for i in e:
                operation = getattr(self, i)

                result = operation(rate)
                if result and MutableBase.debug_mode:
                    print("mutation {}".format(result))

    def __init__(self, raw_dict=None, previous_block = None):

        super(MutableBase, self).__init__(raw_dict,previous_block)

    
