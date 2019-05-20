# -*- coding: utf-8 -*-

from .node import Node
from keras.layers import Flatten, Dropout, BatchNormalization, Activation, Add, Concatenate, Multiply, ZeroPadding2D, Conv2D
from .output import Output, OutCell, OutBlock, Out

class Operation(Node):
    def __init__(self,  raw_dict=None):
        super(Operation, self).__init__(raw_dict=raw_dict)
        

    def build_tensorflow_model(self, model, source1, source2):
        pass

    def build(self, input, neighbour=None):
        if type(input) is OutCell or type(input) is OutBlock or type(input) is Out:
            return input.content
        else:
            return input

    @staticmethod
    def parse_feature_model(feature_model):
        operation = feature_model.get("children")[0] 
        operation_type = Node.get_type(operation)
        operation_element = None

        if operation_type=="void":
            operation_element = Void(operation)
        elif operation_type=="flatten":
            operation_element = Flat(operation)
        elif operation_type=="dropout":

            _value = None
            for child in operation.get("children"):
                element_type = Node.get_type(child)
                if(element_type == "value"):
                    if(len(child.get("children"))):
                        _value = Node.get_type(child.get("children")[0])
            operation_element = Drop(_value=_value, raw_dict=operation)

        elif operation_type=="padding":
            _fillValue = None
            _fillSize = None

            for child in operation.get("children"):
                element_type = Node.get_type(child)
                if(element_type == "fillValue"):
                    if(len(child.get("children"))):
                        _fillValue = Node.get_type(child.get("children")[0])
                if(element_type == "fillSize"):
                    if(len(child.get("children"))):
                        _fillSize = Node.get_type(child.get("children")[0])
                        _fillSize = _fillSize.split("x")
                        if len(_fillSize)==2:
                            _fillSize = tuple(_fillSize)
                        else:
                            _fillSize = None

            #operation_element = Void(operation)
            operation_element = Padding(_fillValue=_fillValue,_fillSize=_fillSize, raw_dict=operation)

        elif operation_type=="batchnormalization":
            _axis = None
            for child in operation.get("children"):
                element_type = Node.get_type(child)
                if(element_type == "axis"):
                    if(len(child.get("children"))):
                        _axis = Node.get_type(child.get("children")[0])

            operation_element = BatchNorm(_axis=_axis, raw_dict=operation)

        elif operation_type=="activation":
            #operation_element = Void(operation)
            operation_element = Active(raw_dict=operation)

        return operation_element
        
class Flat(Operation):
    def __init__(self, raw_dict=None):
        super(Flat, self).__init__(raw_dict=raw_dict)

    def build(self,input):
        input = super(Flat, self).build(input)
        if(input.shape.ndims > 2):
            return Flatten()(input)
        return input

class Void(Operation):
    def __init__(self, raw_dict=None):
        super(Void, self).__init__(raw_dict=raw_dict)

    def build(self,input):
        return input

class Drop(Operation):
    def __init__(self, _value, raw_dict=None):
        super(Drop, self).__init__(raw_dict=raw_dict)
        if not _value:
            self._value =_value = 0.5
            #self.append_parameter("_value","__float__")
        else:
            self._value = int(_value)/10

    def build(self,input):
        input = super(Drop, self).build(input)
        return Dropout(self._value)(input)

class Padding(Operation):
    def __init__(self, _fillValue=None, _fillSize=None, raw_dict=None):
        super(Padding, self).__init__(raw_dict=raw_dict)

        _fillValue = 0
        
        if _fillValue==None:
            self.append_parameter("_fillValue","__int__")
        else:
            self._fillValue = int(_fillValue)

        if not _fillSize:
            self._fillSize = (1,1)
            #self.append_parameter("_fillSize","(__int__,__int__)")
        else:
            self._fillSize = (int(_fillSize[0]), int(_fillSize[1]))

    def build(self,input):
        input = super(Padding, self).build(input)
        if input.shape.ndims ==4:
            return ZeroPadding2D(self._fillSize)(input)
        return input

class BatchNorm(Operation):
    def __init__(self, _axis=None, raw_dict=None):
        super(BatchNorm, self).__init__(raw_dict=raw_dict)
        _axis = 1
        if not _axis:
            self.append_parameter("_axis","__int__")
        else:
            self._axis = int(_axis)

    def build(self,input):
        input = super(BatchNorm, self).build(input)
        return BatchNormalization(axis = self._axis)(input)

class Active(Operation):
    def __init__(self, _method=None, raw_dict=None):
        super(Active, self).__init__(raw_dict=raw_dict)

        _method = "relu"
        activationAcceptedValues = ("tanh","relu","sigmoid","softmax")
        if not _method or str(_method) not in activationAcceptedValues:
            self.append_parameter("_method",'|'.join(str(i) for i in activationAcceptedValues))
        else:
            self._method = str(_method)


class Combination(Node):
    def __init__(self, raw_dict=None):
        super(Combination, self).__init__(raw_dict=raw_dict)

    def build(self, source1, source2):
        pass

    @staticmethod
    def parse_feature_model(feature_model):
        operation = feature_model.get("children")[0] 
        operation_type = Node.get_type(operation)
        operation_element = None

        if operation_type=="sum":
            operation_element = Sum(operation)
        elif operation_type=="product":
            operation_element = Product(operation)
        elif operation_type=="concat":
            _axis = None
            for child in operation.get("children"):
                element_type = Node.get_type(child)
                if(element_type == "axis"):
                    if(len(child.get("children"))):
                        _axis = Node.get_type(child.get("children")[0])

            #operation_element = Sum(operation)
            operation_element = Concat(_axis=_axis, raw_dict=operation)
        
        return operation_element 


class Sum(Combination):
    def __init__(self,  raw_dict=None):
        super(Sum, self).__init__(raw_dict=raw_dict)

    def build(self, source1, source2):
        if source1.shape.dims == source2.shape.dims:
            return Add()([source1, source2])
        return source1
        
class Concat(Combination):
    def __init__(self, _axis=None, raw_dict=None):
        super(Concat, self).__init__(raw_dict=raw_dict)
        _axis = 1
        if not _axis:
            self.append_parameter("_axis","__int__")
        else:
            self._axis = int(_axis)

    def build(self, source1, source2):
        if source1.shape.dims == source2.shape.dims:
            return Concatenate(axis=self._axis)([source1, source2])
        return source1

class Product(Combination):
    def __init__(self, raw_dict=None):
        super(Product, self).__init__(raw_dict=raw_dict)

    
    def build(self, source1, source2):
        return Multiply()([source1, source2])
        