# -*- coding: utf-8 -*-

from .node import Node
from keras import backend as K
from keras.layers import Dense, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D 
from .output import Output, OutCell, OutBlock, Out

class Input(Node):
    def __init__(self, raw_dict=None):
        
        self.raw_dict = raw_dict
        super(Input, self).__init__(raw_dict=raw_dict)

    def build_tensorflow_model(self, model, source1, source2):
        pass

    def build(self, input, neighbour=None):
        if type(input) is OutCell or type(input) is OutBlock or type(input) is Out:
            return input.content
        else:
            return input

    @staticmethod
    def parse_feature_model(feature_model):

        input = feature_model.get("children")[0] 
        input_type = Node.get_type(input)
        input_element = None

        if(input_type=="zeros"):
            input_element = ZerosInput(raw_dict=input)
        elif(input_type=="identity"):
            input_element = IdentityInput(raw_dict=input)
        elif(input_type=="dense"):
            activation = None
            features = None

            for child in input.get("children"):
                element_type = Node.get_type(child)
                if(element_type == "activation"):
                    if(len(child.get("children"))):
                        activation = Node.get_type(child.get("children")[0])

                elif(element_type == "features"):
                    if(len(child.get("children"))):
                        features = Node.get_type(child.get("children")[0])

            input_element = DenseInput(_features=features,_activation=activation,raw_dict=input)

        elif(input_type=="pooling"):
            _kernel = None
            _stride = None
            _type = None
            _padding = None

            for child in input.get("children"):
                element_type = Node.get_type(child)
                if(element_type == "padding"):
                    if(len(child.get("children"))):
                        _padding = Node.get_type(child.get("children")[0])

                elif(element_type == "type"):
                    if(len(child.get("children"))):
                        _type = Node.get_type(child.get("children")[0])

                elif(element_type == "stride"):
                    if(len(child.get("children"))):
                        _stride = Node.get_type(child.get("children")[0])
                        _stride = _stride.split("x")
                        if len(_stride)==2:
                            _stride = tuple(_stride)
                        else:
                            _stride = None

                elif(element_type == "kernel"):
                    if(len(child.get("children"))):
                        _kernel = Node.get_type(child.get("children")[0])
                        _kernel = _kernel.split("x")
                        if len(_kernel)==2:
                            _kernel = tuple(_kernel)
                        else:
                            _kernel = None


            input_element = PoolingInput(_kernel=_kernel,_stride=_stride, _type=_type, _padding=_padding,raw_dict=input)

        elif(input_type=="convolution"):
            _kernel = None
            _stride = None
            _type = None
            _padding = None
            _activation = None
            _features = None

            for child in input.get("children"):
                element_type = Node.get_type(child)
                if(element_type == "padding"):
                    if(len(child.get("children"))):
                        _padding = Node.get_type(child.get("children")[0])

                elif(element_type == "stride"):
                    if(len(child.get("children"))):
                        _stride = Node.get_type(child.get("children")[0])
                        _stride = _stride.split("x")
                        if len(_stride)==2:
                            _stride = tuple(_stride)
                        else:
                            _stride = None

                elif(element_type == "kernel"):
                    if(len(child.get("children"))):
                        _kernel = Node.get_type(child.get("children")[0])
                        _kernel = _kernel.split("x")
                        if len(_kernel)==2:
                            _kernel = tuple(_kernel)
                        else:
                            _kernel = None

                elif(element_type == "activation"):
                    if(len(child.get("children"))):
                        _activation = Node.get_type(child.get("children")[0])

                elif(element_type == "features"):
                    if(len(child.get("children"))):
                        _features = Node.get_type(child.get("children")[0])


            input_element = ConvolutionInput(_kernel=_kernel,_stride=_stride, _padding=_padding,_activation=_activation,_features=_features,raw_dict=input)
        
        return input_element

class ZerosInput(Input):
    def __init__(self, raw_dict=None):
        super(ZerosInput, self).__init__(raw_dict=raw_dict)
        
    def build(self, input, neighbour=None):
        shape = neighbour.shape
        return K.zeros(shape)

class IdentityInput(Input):
    def __init__(self, raw_dict=None):
        super(IdentityInput, self).__init__(raw_dict=raw_dict)

    def build(self, input, neighbour=None):
        return input

class DenseInput(Input):
    def __init__(self, _features, _activation, raw_dict=None):
        super(DenseInput, self).__init__(raw_dict=raw_dict)

        activationAcceptedValues = ("tanh","relu","sigmoid","softmax")
        if not _features:
            self.append_parameter("_features","__int__")
        else:
            self._features = min(512,int(_features))

        if not _activation or str(_activation) not in activationAcceptedValues:
            self.append_parameter("_activation",'|'.join(str(i) for i in activationAcceptedValues))
        else:
            self._activation = str(_activation)

    def build(self, input, neighbour=None):
        input = super(DenseInput, self).build(input)
        return Dense(self._features, activation=self._activation, name=Node.get_name(self))(input.content if hasattr(input,"content") else input)


class PoolingInput(Input):
    def __init__(self, _kernel, _stride, _type, _padding, raw_dict=None):
        super(PoolingInput, self).__init__(raw_dict=raw_dict)

        typeAcceptedValues = ("max","average","dilated","global")
        paddingAcceptedValues = ("valid", "same")

        if not _type or str(_type) not in typeAcceptedValues:
            self.append_parameter("_type",'|'.join(str(i) for i in typeAcceptedValues))
        else:
            self._type = _type

        if self._type !="global":

            if not _kernel:
                self.append_parameter("_kernel","(__int__,__int__)")
            else:
                self._kernel =(min(int(_kernel[0]),3),min(int(_kernel[1]),3))

            if not _stride:
                self.append_parameter("_stride",'(__int__,__int__)')
            else:
                self._stride = (int(_stride[0]), int(_stride[1]))

            if not _padding or str(_padding) not in paddingAcceptedValues:
                self.append_parameter("_padding",'|'.join(str(i) for i in paddingAcceptedValues))
            else:
                self._padding = _padding

    def build(self, input, neighbour=None):
        
        input = super(PoolingInput, self).build(input)
        input = input.content if hasattr(input,"content") else input

        if input.shape.ndims==4:
            if self._type=="max":
                if(self._padding=="valid" and (self._kernel[0]>input.shape.dims[1].value or self._kernel[1]>input.shape.dims[2].value)):
                    self._padding="same"
                return MaxPooling2D(pool_size=self._kernel, strides = self._stride, padding=self._padding, name=Node.get_name(self))(input)
            if self._type=="average":
                if(self._padding=="valid" and (self._kernel[0]>input.shape.dims[1].value or self._kernel[1]>input.shape.dims[2].value)):
                    self._padding="same"
                return AveragePooling2D(pool_size=self._kernel, strides = self._stride, padding=self._padding, name=Node.get_name(self))(input)
        if self._type=="global":
            return input
            #return GlobalMaxPooling2D(name=Node.get_name(self))(input)

        return input

class ConvolutionInput(Input):
    def __init__(self, _kernel, _stride, _features, _padding, _activation, raw_dict=None):
        super(ConvolutionInput, self).__init__(raw_dict=raw_dict)

        activationAcceptedValues = ("tanh","relu","sigmoid","softmax")
        paddingAcceptedValues = ("valid","same")

        if not _kernel:
            self.append_parameter("_kernel","(__int__,__int__)")
        else:
            self._kernel =(min(int(_kernel[0]),3),min(int(_kernel[1]),3))

        if not _stride:
            self.append_parameter("_stride",'(__int__,__int__)')
        else:
            self._stride = (int(_stride[0]), int(_stride[1]))

        if not _features:
            self.append_parameter("_features","__int__")
        else:
            self._features = min(128,int(_features))

        if not _activation or str(_activation) not in activationAcceptedValues:
            self.append_parameter("_activation",'|'.join(str(i) for i in activationAcceptedValues))
        else:
            self._activation = str(_activation)

        if not _padding or str(_padding) not in paddingAcceptedValues:
            self.append_parameter("_padding",'|'.join(str(i) for i in paddingAcceptedValues))
        else:
            self._padding = _padding
        
    def build(self, input, neighbour=None):
        input = super(ConvolutionInput, self).build(input)
        input = input.content if hasattr(input,"content") else input
        
        if input.shape.ndims==4:
            if(self._padding=="valid" and (self._kernel[0]>input.shape.dims[1].value or self._kernel[1]>input.shape.dims[2].value)):
                    self._padding="same"
            return Conv2D(self._features, self._kernel, strides = self._stride, padding=self._padding, activation=self._activation, name=Node.get_name(self))(input)
        return input
        