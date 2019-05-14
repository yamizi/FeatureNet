from .mutable_base import MutableBase

class MutableBlock(MutableBase):
    def __init__(self, raw_dict=None, previous_block = None):

        super(MutableBlock, self).__init__(raw_dict,previous_block)
