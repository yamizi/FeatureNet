from .mutable_base import MutableBase
from model.cell import Cell
from numpy.random import choice


class MutableInput(MutableBase):

    attributes = {"strides_values":"_stride","features_values":"_features", "kernel_values":"_kernel", "type_values":"_type"}
    strides_values = ("1x1","2x2")
    kernel_values = ("1x1","3x1","1x3","3x3","5x1","1x5","5x5","7x1","1x7","7x7",)
    type_values = ("max","average","global")
    features_values = (None,8,16,32,64,128,256, 512, 1024, 2048)
    

    def __init__(self, raw_dict=None, stride=1, features=0):

        self.mutation_operators = (("mutate_type",0.2),("mutate_attributes",0.2))
        super(MutableInput, self).__init__(raw_dict ,stride, features)


    def mutate_type(self):
        from model.input import ZerosInput, DenseInput, IdentityInput, PoolingInput, ConvolutionInput
        inputs = (ZerosInput, DenseInput, IdentityInput, PoolingInput, ConvolutionInput)
        input = choice(inputs, None)()

        #copy previous input attributes
        input.parent_cell = self.parent_cell
        for e in self.attributes.values():
            setattr(input,e, getattr(self,e))

        if self.parent_cell.input1 == self:
            self.parent_cell.input1 = input

        else:
            self.parent_cell.input2 = input


    def mutate_attributes(self):
        attribute_to_mutate = getattr(self,choice(self.attributes.keys(), None))
        attribute_value = choice(attribute_to_mutate, None)
        setattr(self, self.attributes[attribute_to_mutate],attribute_value)
