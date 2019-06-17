from .mutable_base import MutableBase
from numpy.random import choice


class MutableOutput(MutableBase):

    attributes = {"cell_index_values":"_relativeCellIndex"}
    cell_index_values = (1, 2, 3)
    

    def __init__(self, raw_dict=None, stride=1, features=0):

        self.mutation_operators = (("mutate_type",0.5),("mutate_attributes",0.5))
        super(MutableOutput, self).__init__()


    def mutate_type(self):
        from model.output import OutBlock, OutCell
        outputs = (OutBlock, OutCell)
        output = choice(outputs, None)()

        #copy previous output attributes
        output.parent_cell = self.parent_cell
        for e in self.attributes.values():
            setattr(output,e, getattr(self,e))

        self.parent_cell.output = output
        return ("mutate_output_type",output )


    def mutate_attributes(self):
        attribute_to_mutate = choice(list(self.attributes.keys()), None)
        attribute_value = choice(getattr(self,attribute_to_mutate), None)
        setattr(self, self.attributes[attribute_to_mutate],attribute_value)
        return ("mutate_output_attribute",attribute_to_mutate, attribute_value )
