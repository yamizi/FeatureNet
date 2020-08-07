from .mutable_base import MutableBase, MutationStrategies
from numpy.random import choice, rand

class MutableInput(MutableBase):

    strides_values = ((1,1),(2,2))
    kernel_values = ((1,1),(3,1),(1,3),(3,3),(5,1),(1,5),(5,5),(7,1),(1,7),(7,7))
    pool_type_values = ("max","average","global")
    conv_type_values = ("normal","separable","depthwise")
    activation_values = ("relu","sigmoid",None)
    features_values = (None,8,16,32,64,128,256, 512, 1024, 2048)
    mutation_operators = ("mutate_type", 0), ("mutate_attributes", 1)
    type_inputs = [("identity",0.25),("dense",0.25),("pooling",0.25),("convolution",0.25),("embedding",0),("lstm",0)]

    def __init__(self, raw_dict=None, stride=1, features=0):

        self.mutation_operators = MutableInput.mutation_operators
        from model.input import DenseInput, IdentityInput, PoolingInput, ConvolutionInput, EmbeddingInput, \
            LSTMInput

        self.available_inputs = {"identity":IdentityInput,"dense":DenseInput,"pooling":PoolingInput,"convolution":ConvolutionInput,"embedding":EmbeddingInput,"lstm":LSTMInput}

        # mapping between the possible values and the actual node property
        self.attributes = {"strides_values": "_stride", "features_values": "_features", "kernel_values": "_kernel",
                      "pool_type_values": "_pool_type", "conv_type_values": "_conv_type",
                      "activation_values": "_activation"}

        super(MutableInput, self).__init__()

    def mutate_type(self, rate=1):
        prob = rand()
        if prob < rate or MutableBase.mutation_stategy==MutationStrategies.CHOICE:

            inputs = MutableInput.type_inputs
            e, p = zip(*inputs)

            #We only allow second input to be zero
            if self.parent_cell.input2 == self:
                from model.input import ZerosInput
                e.append("zero")
                p = None
                inputs["zero"] = ZerosInput

            selected = getattr(self.available_inputs, choice(e, "identity", p))
            if selected is not None:
                input = selected()

            #copy previous input attributes
            input.parent_cell = self.parent_cell
            for e in self.attributes.values():
                setattr(input,e, getattr(self,e,None))

            if self.parent_cell.input1 == self:
                self.parent_cell.input1 = input

            else:
                self.parent_cell.input2 = input

            return ("mutate_input_type",input)
        else:
            return ("mutate_input_type",)

    def mutate_attributes(self, rate=1):
        attrs = []
        if MutableBase.mutation_stategy==MutationStrategies.CHOICE:
            attribute_to_mutate =choice(list(self.attributes.keys()), None)
            attr = getattr(self,attribute_to_mutate, None)
            if attr is not None:
                attribute_value = attr[choice( len(attr))]
                attribute = self.attributes[attribute_to_mutate]
                attr_exists = getattr(self,attribute, False)
                if attr_exists != False:
                    setattr(self,attribute ,attribute_value)
                    attrs = [("mutate_input_attribute",self.attributes[attribute_to_mutate], attr_exists, attribute_value )]
        else:
            for attribute_to_mutate in self.attributes.keys():
                prob = rand()
                if prob < rate:
                    attr = getattr(self,attribute_to_mutate)
                    attribute_value = attr[choice( len(attr))]
                    setattr(self, self.attributes[attribute_to_mutate],attribute_value)

                    attrs.append(("mutate_input_attribute",self.attributes[attribute_to_mutate], attribute_value ))
        return attrs
