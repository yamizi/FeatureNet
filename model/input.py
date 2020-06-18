# -*- coding: utf-8 -*-

from .node import Node
from keras import backend as K
from keras.layers import Embedding
from keras.layers import Dense, Conv2D, SeparableConv2D, DepthwiseConv2D, Conv1D, SeparableConv1D
from keras.layers import AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D 
from .output import Output, OutCell, OutBlock, Out

from .mutation.mutable_input import MutableInput

class Input(MutableInput, Node):

    min_features = 6
    max_features = 2048
    
    def __init__(self, raw_dict=None, stride=1, features=0, cell=None):
        
        self.raw_dict = raw_dict
        self._stride = stride
        self._features = features
        self._relative_features = None
        self.build_raw = False
        self.last_build = None

        self.parent_cell = cell
        super(Input, self).__init__(raw_dict=raw_dict)

    def set_stride(self,stride):
        stride_x, stride_y = int(stride[0]), int(stride[1])
        self._stride = (max(1,min(stride_x,2)), max(1,min(stride_y,2)))

    def set_features(self,features, relative_features=False):
        if relative_features:
            self._relative_features = float(features)
        else:
            self._features =  max(Input.min_features,min(int(features),Input.max_features))

    def build(self, input, neighbour=None):
        input =  input.content if hasattr(input,"content") and input.content is not None else input
        input_features = input.shape.as_list()[-1]
        if self._relative_features:
            self.set_features(input_features * self._relative_features)

        if not self._features:
            self._features = Input.min_features
            
        return input
        
    @staticmethod
    def parse_feature_model(feature_model):

        input = feature_model.get("children")[0] 
        input_type = Node.get_type(input)
        input_element = None
        build_raw = False

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
                elif (element_type == "build_raw"):
                    if (len(child.get("children"))):
                        build_raw = Node.get_type(child.get("build_raw")[0])
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
                elif (element_type == "build_raw"):
                    if (len(child.get("children"))):
                        build_raw = Node.get_type(child.get("build_raw")[0])

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
#_input_dim, _input_length,
        elif (input_type == "embedding"):
            params = dict(_input_dim = None,
            _input_length = None,
            _dropout = 0.2,
            _embed_dim = 128)

            for child in input.get("children"):
                element_type = Node.get_type(child)
                _element_type = "_{}".format(element_type)
                if (_element_type in params):
                    if (len(child.get("children"))):
                        params[_element_type] = Node.get_type(child.get("children")[0])

            input_element = EmbeddingInput(**params,raw_dict=input)

        elif(input_type=="convolution"):
            _kernel = None
            _stride = None
            _type = None
            _padding = None
            _activation = "relu"
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

                elif(element_type == "type"):
                    if(len(child.get("children"))):
                        _type = Node.get_type(child.get("children")[0])

                elif(element_type == "features"):
                    if(len(child.get("children"))):
                        _features = Node.get_type(child.get("children")[0])

                elif (element_type == "build_raw"):
                    if (len(child.get("children"))):
                        build_raw = Node.get_type(child.get("build_raw")[0])

            input_element = ConvolutionInput(_kernel=_kernel,_stride=_stride, _padding=_padding,_activation=_activation,_features=_features, _type=_type, raw_dict=input)

        input_element.build_raw = build_raw
        return input_element

class ZerosInput(Input):
    def __init__(self, raw_dict=None, cell=None):
        super(ZerosInput, self).__init__(raw_dict=raw_dict, cell=cell)
        
    def build(self, input, neighbour=None):
        shape = neighbour.shape if neighbour else input.shape
        return K.zeros(shape)

class IdentityInput(Input):
    def __init__(self, raw_dict=None, cell=None):
        super(IdentityInput, self).__init__(raw_dict=raw_dict, cell=cell)

    def build(self, input, neighbour=None):
        return input


class EmbeddingInput(Input):
    def __init__(self, _input_dim, _input_length, _dropout = 0.2, _embed_dim = 128, raw_dict=None, cell=None):
        super(IdentityInput, self).__init__(raw_dict=raw_dict, cell=cell)

        self._input_dim = _input_dim
        self._input_length = _input_length
        self._dropout = _dropout
        self. _embed_dim = _embed_dim

    def build(self, input, neighbour=None):
        input = super(EmbeddingInput, self).build(input)
        self.last_build = Embedding(self._input_dim, self._embed_dim,self._input_length, dropout = self._dropout)

        if self.build_raw:
            return input
        return self.last_build(input)

class DenseInput(Input):
    def __init__(self, _features=128, _activation="relu", raw_dict=None, cell=None):
        super(DenseInput, self).__init__(raw_dict=raw_dict, cell=cell)
        activationAcceptedValues = ("tanh","relu","sigmoid","softmax", "none")
        if not _features:
            self.append_parameter("_features","__int__")
        else:
            self._features = int(_features)

        if not _activation or str(_activation) not in activationAcceptedValues:
            self.append_parameter("_activation",'|'.join(str(i) for i in activationAcceptedValues))
        else:
            self._activation = str(_activation) if _activation !="none" else None

    def build(self, input, neighbour=None):
        input = super(DenseInput, self).build(input)
        self.last_build =  Dense(self._features, activation=self._activation, name=Node.get_name(self))

        if self.build_raw:
            return input
        return self.last_build(input)

class PoolingInput(Input):
    def __init__(self, _kernel=(3,3), _stride=(1,1), _type="max", _padding="same", raw_dict=None, cell=None):
        super(PoolingInput, self).__init__(raw_dict=raw_dict, cell=cell)

        typeAcceptedValues = ("max","average","global")
        paddingAcceptedValues = ("same",)# ("valid", "same")

        if not _type or str(_type) not in typeAcceptedValues:
            self.append_parameter("_type",'|'.join(str(i) for i in typeAcceptedValues))
            self._type = "max"
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
                self._padding = paddingAcceptedValues[0]
            else:
                self._padding = _padding

    def build(self, input, neighbour=None):
        
        input = super(PoolingInput, self).build(input)
        input = input.content if hasattr(input,"content") else input

        if input.shape.ndims==4:
            if self._type=="max":
                if(self._padding=="valid" and (self._kernel[0]>input.shape.dims[1].value or self._kernel[1]>input.shape.dims[2].value)):
                    self._padding="same"
                self.last_build =  MaxPooling2D(pool_size=self._kernel, strides = self._stride, padding=self._padding, name=Node.get_name(self))
            if self._type=="average":
                if(self._padding=="valid" and (self._kernel[0]>input.shape.dims[1].value or self._kernel[1]>input.shape.dims[2].value)):
                    self._padding="same"
                self.last_build = AveragePooling2D(pool_size=self._kernel, strides = self._stride, padding=self._padding, name=Node.get_name(self))
            if self._type=="global":
                #return input
                self.last_build =  GlobalAveragePooling2D(name=Node.get_name(self))

            if self.build_raw:
                return input
            return self.last_build(input)

        return input

class ConvolutionInput(Input):
    def __init__(self, _kernel=(3,3), _stride=(1,1), _features=8, _padding="same", _activation="relu", _type="normal", raw_dict=None, cell=None):
        super(ConvolutionInput, self).__init__(raw_dict=raw_dict, cell=cell)

        activationAcceptedValues = ("tanh","relu","sigmoid","softmax", "none")
        paddingAcceptedValues = ("same",) #("valid","same")
        typeAcceptedValues = ("normal","separable","depthwise")

        if not _type or str(_type) not in typeAcceptedValues:
            self.append_parameter("_type",'|'.join(str(i) for i in typeAcceptedValues))
            self._type = "normal"
        else:
            self._type = _type

        if not _kernel:
            self.append_parameter("_kernel","(__int__,__int__)")
        else:
            self._kernel =(min(int(_kernel[0]),5),min(int(_kernel[1]),5))

        if not _stride:
            self.append_parameter("_stride",'(__int__,__int__)')
        else:
            self._stride = (int(_stride[0]), int(_stride[1]))

        if not _features:
            self.append_parameter("_features","__int__")
        else:
            self._features =int(_features)

        if not _activation or str(_activation) not in activationAcceptedValues:
            self.append_parameter("_activation",'|'.join(str(i) for i in activationAcceptedValues))
        else:
            self._activation = str(_activation) if _activation != "none" else None

        if not _padding or str(_padding) not in paddingAcceptedValues:
            self.append_parameter("_padding",'|'.join(str(i) for i in paddingAcceptedValues))
            self._padding = paddingAcceptedValues[0]
        else:
            self._padding = _padding
        
    def build(self, input, neighbour=None):
        input = super(ConvolutionInput, self).build(input)
        input = input.content if hasattr(input,"content") else input
        self.last_build = None

        if input.shape.ndims==4:
            if self._type == "normal":
                if(self._padding=="valid" and (self._kernel[0]>input.shape.dims[1].value or self._kernel[1]>input.shape.dims[2].value)):
                        self._padding="same"
                self.last_build= Conv2D(self._features, self._kernel, strides = self._stride, padding=self._padding, activation=self._activation, name=Node.get_name(self))
            elif self._type == "separable":
                self.last_build= SeparableConv2D(self._features, self._kernel, strides = self._stride, padding=self._padding, activation=self._activation, name=Node.get_name(self))
            elif self._type == "depthwise":
                self.last_build= DepthwiseConv2D(self._kernel, strides = self._stride, padding=self._padding, activation=self._activation, name=Node.get_name(self))
        
        elif input.shape.ndims==3:
            if self._type == "normal":
                if(self._padding=="valid" and (self._kernel[0]>input.shape.dims[1].value or self._kernel[1]>input.shape.dims[2].value)):
                        self._padding="same"
                self.last_build= Conv1D(self._features, self._kernel[0], strides = self._stride[0], padding=self._padding, activation=self._activation, name=Node.get_name(self))
            elif self._type == "separable":
                self.last_build= SeparableConv1D(self._features, self._kernel[0], strides = self._stride[0], padding=self._padding, activation=self._activation, name=Node.get_name(self))


        if self.last_build is not None:
            if self.build_raw:
                return input
            return self.last_build(input)
        return input
