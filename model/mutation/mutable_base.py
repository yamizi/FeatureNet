from enum import Enum
from numpy.random import choice

class MutationStrategies(Enum):
    CHOICE = 1 # at every node choose only one of the children to mutate
    ALL = 2 # at every node mutate all the children according to mutation rate

class SelectionStrategies(Enum):
    ELITIST = 1 # choose top N models in terms of accuracy
    HYBRID = 2 # chose top N/4 models + 3N/4 randomly selected according to accuracy distribution
    PARETO = 3 # hybrid within the PARETO models

class MutableBase(object):
  
    mutation_operators = []
    debug_mode = True
    mutation_stategy = MutationStrategies.CHOICE
    MAX_NB_CELLS = 10
    MAX_NB_BLOCKS = 20
    selection_stragey = SelectionStrategies.HYBRID
    
    def mutate(self, rate=1):
        e,p =  zip(*self.mutation_operators)
        result = None
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

        if len(result)==2:
            self.mutations.append({"mutation_type":result[0],"mutation_target":result[1].__class__.__name__})
        elif len(result)==3:
            self.mutations.append({"mutation_type":result[0],"mutation_target":"{}#{}".format(result[1],result[2])})


    def __init__(self, raw_dict=None, previous_block = None):

        self.mutations = []
        super(MutableBase, self).__init__(raw_dict,previous_block)

    
