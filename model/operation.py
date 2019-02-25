# -*- coding: utf-8 -*-

from .node import Node

class Operation(Node):
    def __init__(self,  raw_dict=None):
        super(Operation, self).__init__(raw_dict=raw_dict)
        

    def build_tensorflow_model(self, model, source1, source2):
        pass

    @staticmethod
    def parse_feature_model(feature_model):
        operation = feature_model.get("children")[0] 
        operation_type = Node.get_type(operation)
        operation_element = None

        if operation_type=="void":
            operation_element = Void(operation)
        elif operation_type=="flatten":
            operation_element = Flatten(operation)
        elif operation_type=="dropout":

            _value = None
            for child in operation.get("children"):
                element_type = Node.get_type(child)
                if(element_type == "value"):
                    if(len(child.get("children"))):
                        _value = Node.get_type(child.get("children")[0])
            operation_element = Dropout(_value=_value, raw_dict=operation)

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
            operation_element = Padding(_fillValue=_fillValue,_fillSize=_fillSize, raw_dict=operation)

        elif operation_type=="batchnormalization":
            _axis = None
            for child in operation.get("children"):
                element_type = Node.get_type(child)
                if(element_type == "axis"):
                    if(len(child.get("children"))):
                        _axis = Node.get_type(child.get("children")[0])

            operation_element = BatchNormalization(_axis=_axis, raw_dict=operation)

        elif operation_type=="activation":
            operation_element = Activation(raw_dict=operation)

        return operation_element
        
class Flatten(Operation):
    def __init__(self, raw_dict=None):
        super(Flatten, self).__init__(raw_dict=raw_dict)

class Void(Operation):
    def __init__(self, raw_dict=None):
        super(Void, self).__init__(raw_dict=raw_dict)

class Dropout(Operation):
    def __init__(self, _value, raw_dict=None):
        super(Dropout, self).__init__(raw_dict=raw_dict)

        if not _value:
            self.append_parameter("_value","__int__")
        else:
            self._value = int(_value)

class Padding(Operation):
    def __init__(self, _fillValue=None, _fillSize=None, raw_dict=None):
        super(Padding, self).__init__(raw_dict=raw_dict)
        if not _fillValue:
            self.append_parameter("_fillValue","__int__")
        else:
            self._fillValue = int(_fillValue)

        if not _fillSize:
            self.append_parameter("_fillSize","(__int__,__int__)")
        else:
            self._fillSize = (int(_fillSize[0]), int(_fillSize[1]))

class BatchNormalization(Operation):
    def __init__(self, _axis=None, raw_dict=None):
        super(BatchNormalization, self).__init__(raw_dict=raw_dict)

        if not _axis:
            self.append_parameter("_axis","__int__")
        else:
            self._axis = int(_axis)

class Activation(Operation):
    def __init__(self, _method=None, raw_dict=None):
        super(Activation, self).__init__(raw_dict=raw_dict)

        activationAcceptedValues = ("tanh","relu","sigmoid","softmax")
        if not _method or str(_method) not in activationAcceptedValues:
            self.append_parameter("_method",'|'.join(str(i) for i in activationAcceptedValues))
        else:
            self._method = str(_method)


class Combination(Node):
    def __init__(self, raw_dict=None):
        super(Combination, self).__init__(raw_dict=raw_dict)
        

    def build_tensorflow_model(self, model, source1, source2):
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

            operation_element = Concat(_axis=_axis, raw_dict=operation)
        
        return operation_element 


class Sum(Combination):
    def __init__(self,  raw_dict=None):
        super(Sum, self).__init__(raw_dict=raw_dict)
        
class Concat(Combination):
    def __init__(self, _axis=None, raw_dict=None):
        super(Concat, self).__init__(raw_dict=raw_dict)
        if not _axis:
            self.append_parameter("_axis","__int__")
        else:
            self._axis = int(_axis)

class Product(Combination):
    def __init__(self, raw_dict=None):
        super(Product, self).__init__(raw_dict=raw_dict)
        