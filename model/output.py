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
        
        if output_type=="block":
            _relativeBlockIndex = None
            for child in output.get("children"):
                element_type = Node.get_type(child)
                if(element_type == "relativeBlockIndex"):
                    if(len(child.get("children"))):
                        _relativeBlockIndex = Node.get_type(child.get("children")[0])
            output_element = OutBlock(_relativeBlockIndex=_relativeBlockIndex, raw_dict=output)

        elif output_type=="cell":
            _relativeCellIndex = None
            for child in output.get("children"):
                element_type = Node.get_type(child)
                if(element_type == "relativeCellIndex"):
                    if(len(child.get("children"))):
                        _relativeCellIndex = Node.get_type(child.get("children")[0])
            output_element = OutCell(_relativeCellIndex=_relativeCellIndex, raw_dict=output)

        else:
            output_element = Out(output)
        
        return output_element


class Out(Output):
    def __init__(self,  raw_dict=None):
        super(Out, self).__init__(raw_dict=raw_dict)
        
class OutBlock(Output):
    def __init__(self, _relativeBlockIndex=None,  raw_dict=None):
        super(OutBlock, self).__init__(raw_dict=raw_dict)
        if not _relativeBlockIndex:
            self._relativeBlockIndex = 1
        else:
            self._relativeBlockIndex = int(_relativeBlockIndex)

        self.currentIndex = self._relativeBlockIndex


class OutCell(Output):
    def __init__(self, _relativeCellIndex=None, raw_dict=None):
        super(OutCell, self).__init__(raw_dict=raw_dict)
        if not _relativeCellIndex:
            self._relativeCellIndex = 1
        else:
            self._relativeCellIndex = int(_relativeCellIndex)

        self.currentIndex = self._relativeCellIndex
