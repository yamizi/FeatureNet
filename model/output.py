# -*- coding: utf-8 -*-

from .node import Node

class Output(Node):
    def __init__(self, raw_dict=None):
        super(Output, self).__init__(raw_dict=raw_dict)
        

    def build_tensorflow_model(self, model, source1, source2):
        pass

    @staticmethod
    def parse_feature_model(feature_model):
        output = feature_model.get("children")[0] 
        output_type = Node.get_type(output)
        if output_type=="out":
            output_element = Out(output)
        elif output_type=="block":
            _relativeBlockIndex = None
            _relativeCellIndex = None
            for child in output.get("children"):
                element_type = Node.get_type(child)
                if(element_type == "relativeCellIndex"):
                    if(len(child.get("children"))):
                        _relativeCellIndex = Node.get_type(child.get("children")[0])
                if(element_type == "relativeBlockIndex"):
                    if(len(child.get("children"))):
                        _relativeBlockIndex = Node.get_type(child.get("children")[0])
            output_element = OutBlock(_relativeBlockIndex=_relativeBlockIndex,_relativeCellIndex=_relativeCellIndex, raw_dict=output)

        elif output_type=="cell":
            _relativeCellIndex = None
            for child in output.get("children"):
                element_type = Node.get_type(child)
                if(element_type == "relativeCellIndex"):
                    if(len(child.get("children"))):
                        _relativeCellIndex = Node.get_type(child.get("children")[0])
            output_element = OutCell(_relativeCellIndex=_relativeCellIndex, raw_dict=output)
        
        return output_element


class Out(Output):
    def __init__(self,  raw_dict=None):
        super(Out, self).__init__(raw_dict=raw_dict)
        
class OutBlock(Output):
    def __init__(self, _relativeBlockIndex=None, _relativeCellIndex=None, raw_dict=None):
        super(OutBlock, self).__init__(raw_dict=raw_dict)
        if not _relativeBlockIndex:
            self.append_parameter("_relativeBlockIndex","__int__")
        else:
            self._relativeBlockIndex = int(_relativeBlockIndex)

        if not _relativeCellIndex:
            self.append_parameter("_relativeCellIndex","__int__")
        else:
            self._relativeCellIndex = int(_relativeCellIndex)

class OutCell(Output):
    def __init__(self, _relativeCellIndex=None, raw_dict=None):
        super(OutCell, self).__init__(raw_dict=raw_dict)
        if not _relativeCellIndex:
            self.append_parameter("_relativeCellIndex","__int__")
        else:
            self._relativeCellIndex = int(_relativeCellIndex)
