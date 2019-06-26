from .mutable_base import MutableBase, MutationStrategies
from numpy.random import choice, rand


class MutableOperation(MutableBase):

    attributes = {"dropout_values":"_value", "activation_method":"_method"}
    dropout_values = (0.7, 0.3, 0.5)
    activation_method = ("relu", "tanh", "sigmoid")

    def __init__(self, raw_dict=None, stride=1, features=0):

        self.mutation_operators = (("mutate_type",0.5),("mutate_attributes",0.5))
        super(MutableOperation, self).__init__()


    def mutate_type(self,rate=1):
        prob = rand()
        if prob < rate or MutableBase.mutation_stategy==MutationStrategies.CHOICE:
            from model.operation import Active, BatchNorm, Void, Drop
            operations = (Active, BatchNorm, Void, Drop)
            operation = choice(operations, None)()

            #copy previous operation attributes
            operation.parent_cell = self.parent_cell
            for e in self.attributes.values():
                setattr(operation,e, getattr(self,e))

            if self.parent_cell.operation1 == self:
                self.parent_cell.operation1 = operation

            else:
                self.parent_cell.operation2 = operation

            return ("mutate_operation_type",operation )
            
        return ("mutate_operation_type",)


    def mutate_attributes(self,rate=1):
        attrs = []
        if MutableBase.mutation_stategy==MutationStrategies.CHOICE:
            attribute_to_mutate = choice(list(self.attributes.keys()), None)
            attribute_value = choice(getattr(self,attribute_to_mutate), None)
            setattr(self, self.attributes[attribute_to_mutate],attribute_value)
            attrs =  [("mutate_operation_attribute",attribute_to_mutate, attribute_value )]
        else:
            
            for attribute_to_mutate in self.attributes.keys():
                prob = rand()
                if prob < rate:
                    attribute_value = choice(getattr(self,attribute_to_mutate), None)
                    setattr(self, self.attributes[attribute_to_mutate],attribute_value)
                    attrs.append(("mutate_operation_attribute",attribute_to_mutate, attribute_value ))

        return attrs
