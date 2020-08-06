from .mutable_base import MutableBase, MutationStrategies
from numpy.random import choice, rand


class MutableOutput(MutableBase):

    attributes = {"cell_index_values":"_relativeCellIndex"}
    cell_index_values = (1, 2, 3)
    mutation_operators = (("mutate_type", 0.5), ("mutate_attributes", 0.5))

    def __init__(self, raw_dict=None, stride=1, features=0):

        self.mutation_operators = MutableOutput.mutation_operators
        super(MutableOutput, self).__init__()


    def mutate_type(self,rate=1):
        prob = rand()
        if prob < rate or MutableBase.mutation_stategy==MutationStrategies.CHOICE:
            from model.output import OutBlock, OutCell
            outputs = (OutBlock, OutCell)
            output = choice(outputs, None)()

            #copy previous output attributes
            output.parent_cell = self.parent_cell
            for e in self.attributes.values():
                setattr(output,e, getattr(self,e))

            self.parent_cell.output = output
            return ("mutate_output_type",output )
        return ("mutate_output_type",)


    def mutate_attributes(self,rate=1):
        attrs = []
        if MutableBase.mutation_stategy==MutationStrategies.CHOICE:
            attribute_to_mutate = choice(list(self.attributes.keys()), None)
            attribute_value = choice(getattr(self,attribute_to_mutate), None)
            setattr(self, self.attributes[attribute_to_mutate],attribute_value)
            attrs =  [("mutate_output_attribute",self.attributes[attribute_to_mutate], attribute_value )]
        else:
            
            for attribute_to_mutate in self.attributes.keys():
                prob = rand()
                if prob < rate:
                    attribute_value = choice(getattr(self,attribute_to_mutate), None)
                    setattr(self, self.attributes[attribute_to_mutate],attribute_value)
                    attrs.append(("mutate_output_attribute",self.attributes[attribute_to_mutate], attribute_value ))

        return attrs
