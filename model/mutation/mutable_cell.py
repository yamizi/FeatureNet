from .mutable_base import MutableBase

class MutableCell(MutableBase):
    def __init__(self, raw_dict=None, input1=None, operation1=None, input2=None, operation2=None, output=None, output_combination=None):

        self.mutation_operators = (("mutate_input1",0.4),("mutate_input2",0.3),("mutate_output",0.3))
        super(MutableCell, self).__init__(raw_dict)


    def mutate_input1(self):
        self.input1.mutate()
        return ("cell","input1")

    def mutate_input2(self):
        self.input2.mutate()
        return ("cell","input2")

    def mutate_operation1(self):
        self.operation1.mutate()

    def mutate_operation2(self):
        self.operation2.mutate()

    def mutate_combination(self):
        self.combination.mutate()

    def mutate_output(self):
        self.output.mutate()
        return ("cell","output")

    