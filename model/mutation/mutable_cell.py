from .mutable_base import MutableBase
from model.cell import Cell

class MutableCell(MutableBase):
    def __init__(self, raw_dict=None, input1=None, operation1=None, input2=None, operation2=None, output=None, output_combination=None):

        self.mutation_operators = (("mutate_input1",0.2),("mutate_input2",0.2),("mutate_operation1",0.2),("mutate_operation2",0.2),("mutate_output",0.2),("remove_combination",0.2))
        super(MutableCell, self).__init__(raw_dict,input1,operation1,input2, operation2, output, output_combination)


    def mutate_input1(self):
        self.input1.mutate()

    def mutate_input2(self):
        self.input2.mutate()

    def mutate_operation1(self):
        self.operation1.mutate()

    def mutate_operation2(self):
        self.operation2.mutate()

    def mutate_combination(self):
        self.combination.mutate()

    def mutate_output(self):
        self.output.mutate()

    