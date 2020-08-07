from .mutable_block import MutableBlock
from .mutable_cell import MutableCell
from .mutable_input import MutableInput
from .mutable_operation import MutableOperation
from .mutable_combination import MutableCombination
from .mutable_output import MutableOutput

import json


class MutableParameters(object):

    mutable_classes = {"MutableBlock":MutableBlock, "MutableCell":MutableCell, "MutableInput":MutableInput, "MutableOperation":MutableOperation, "MutableCombination":MutableCombination, "MutableOutput":MutableOutput}

    @staticmethod
    def set_config(config):
        for c, k in config.items():
            cls = MutableParameters.mutable_classes.get(c)
            for attr, v in k.items():
                setattr(cls, attr, v)

    @staticmethod
    def load_config(file_path):
        with open(file_path, "r") as file:
            config = json.load(file)
            if config:
                print("Configuration {} loaded".format(file_path))
            MutableParameters.set_config(config["mutable_parameters"])

            return config


