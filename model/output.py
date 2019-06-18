# -*- coding: utf-8 -*-

from .node import Node

from .mutation.mutable_output import MutableOutput

class Output(MutableOutput, Node):

    currentIndex =0
    _relativeBlockIndex=  None
    _relativeCellIndex = None
    
    def __init__(self, raw_dict=None, cell=None):
        self.parent_cell = cell
        super(Output, self).__init__(raw_dict=raw_dict)

    def __deepcopy__(self, memo):
        newone = type(self)()
        newone._relativeBlockIndex = self._relativeBlockIndex
        newone._relativeCellIndex = self._relativeCellIndex
        newone.parent_cell = self.parent_cell
        return newone

    @property
    def shape(self):
        if self.content is not None:
            return self.content.shape
        return None

    def build(self, input):
        self.content = input.content if hasattr(input,"content") and input.content is not None else input
        return self
        

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
                if(element_type == "relativecellindex"):
                    if(len(child.get("children"))):
                        _relativeCellIndex = Node.get_type(child.get("children")[0])
            output_element = OutCell(_relativeCellIndex=_relativeCellIndex, raw_dict=output)

        else:
            output_element = Out(output)
        
        return output_element


class Out(Output):
    def __init__(self,  raw_dict=None, cell=None):
        super(Out, self).__init__(raw_dict=raw_dict, cell=cell)
        
class OutBlock(Output):
    def __init__(self, _relativeBlockIndex=None,  raw_dict=None, cell=None):
        super(OutBlock, self).__init__(raw_dict=raw_dict, cell=cell)
        if not _relativeBlockIndex:
            self._relativeBlockIndex = 0
        else:
            self._relativeBlockIndex = int(_relativeBlockIndex)

        self.currentIndex = self._relativeBlockIndex

class OutCell(Output):
    
    def __init__(self, _relativeCellIndex=None, raw_dict=None, cell=None):
        super(OutCell, self).__init__(raw_dict=raw_dict, cell=cell)
        #_relativeCellIndex = 0
        if not _relativeCellIndex:
            self._relativeCellIndex = 1
        else:
            self._relativeCellIndex = int(_relativeCellIndex)+1
            #print("{} skips {} cells".format(self.get_name(),self._relativeCellIndex-1))

    def build(self, input):
        super(OutCell, self).build(input)
        self.currentIndex = self._relativeCellIndex
        return self

