from .mutable_block import MutableBlock
from .mutable_cell import MutableCell
from .mutable_input import MutableInput
from .mutable_operation import MutableOperation
from .mutable_combination import MutableCombination
from .mutable_output import MutableOutput

import json


class Mutable_parameters(object):

    mutable_classes = {"MutableBlock":MutableBlock, "MutableCell":MutableCell, "MutableInput":MutableInput, "MutableOperation":MutableOperation, "MutableCombination":MutableCombination, "MutableOutput":MutableOutput}

    @staticmethod
    def set_config(config):
        for c, k in config.items():
            cls = getattr(Mutable_parameters.mutable_classes, c)
            for attr, v in k.items():
                setattr(cls, attr, v)
                print(v, getattr(cls, attr))

    @staticmethod
    def load_config(file_path):
        with open(file_path, "r") as file:
            config = json.load(file)
            Mutable_parameters.set_config(config)


