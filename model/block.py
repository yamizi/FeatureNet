# -*- coding: utf-8 -*-


class Block(object):
    def __init__(self, previous_block = None):

        self.is_root = True
        self.cells = []

        if previous_block:
            self.previous_block = previous_block
            self.is_root = False

    def append_cell(self, cell):
        self.cells.append(cell)


    def build_tensorflow_model(self, model):
        pass


    @staticmethod
    def parse_feature_mode(feature_model):
        print(feature_model)
