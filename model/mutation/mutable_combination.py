from .mutable_base import MutableBase, MutationStrategies
from numpy.random import choice, rand


class MutableCombination(MutableBase):


    def __init__(self, raw_dict=None, stride=1, features=0):

        self.mutation_operators = (("mutate_type",1))
        super(MutableCombination, self).__init__()


    def mutate_type(self,rate=1):
        prob = rand()
        if prob < rate or MutableBase.mutation_stategy==MutationStrategies.CHOICE:
            from model.operation import Concat, Sum
            combinations = (Concat, Sum)
            combination = choice(combinations, None)()

            #copy previous combination attributes
            combination.parent_cell = self.parent_cell
            for e in self.attributes.values():
                setattr(combination,e, getattr(self,e))

            self.parent_cell.combination = combination
            return ("mutate_combination_type",combination )
        return ("mutate_combination_type",)


    