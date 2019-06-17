from .mutable_base import MutableBase
from numpy.random import choice


class MutableInput(MutableBase):

    attributes = {"strides_values":"_stride","features_values":"_features", "kernel_values":"_kernel", "pool_type_values":"_type", "conv_type_values":"_type", "activation_values":"_activation"}
    strides_values = ((1,1),(2,2))
    kernel_values = ((1,1),(3,1),(1,3),(3,3),(5,1),(1,5),(5,5),(7,1),(1,7),(7,7))
    pool_type_values = ("max","average","global")
    conv_type_values = ("normal","separable","depthwise")
    activation_values = ("relu","sigmoid",None)
    features_values = (None,8,16,32,64,128,256, 512, 1024, 2048)
    

    def __init__(self, raw_dict=None, stride=1, features=0):

        self.mutation_operators = (("mutate_type",0.5),("mutate_attributes",0.5))
        super(MutableInput, self).__init__()


    def mutate_type(self):
        from model.input import ZerosInput, DenseInput, IdentityInput, PoolingInput, ConvolutionInput
        inputs = (ZerosInput, DenseInput, IdentityInput, PoolingInput, ConvolutionInput)
        input = choice(inputs, None)()

        #copy previous input attributes
        input.parent_cell = self.parent_cell
        for e in self.attributes.values():
            setattr(input,e, getattr(self,e,None))

        if self.parent_cell.input1 == self:
            self.parent_cell.input1 = input

        else:
            self.parent_cell.input2 = input

        return ("mutate_input_type",input)



    def mutate_attributes(self):
        attribute_to_mutate =choice(list(self.attributes.keys()), None)
        attr = getattr(self,attribute_to_mutate)
        attribute_value = attr[choice( len(attr))]
        setattr(self, self.attributes[attribute_to_mutate],attribute_value)

        return ("mutate_input_attribute",attribute_to_mutate, attribute_value )
