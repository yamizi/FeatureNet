from .input import PoolingInput, ConvolutionInput, DenseInput
from .block import Block
from .cell import Cell

def lenet5_blocks():
    blocks = []

    block1 = Block()
    block1.set_stride("1x1")
    block1.set_features(600)
    cell11 = Cell(input1 = ConvolutionInput((5,5),None,None,"same", "tanh"))
    block1.append_cell(cell11)
    cell12 = Cell(input1 = PoolingInput((2,2),None,"average", "valid"))
    block1.append_cell(cell12)

    block2 = Block()
    block2.set_stride("1x1")
    cell21 = Cell(input1 = ConvolutionInput((5,5),None, 12,"same", "tanh"))
    block2.append_cell(cell21)

    block22 = Block()
    block22.set_stride("2x2")
    cell22 = Cell(input1 = PoolingInput((2,2),None,"average", "valid"))
    block22.append_cell(cell22)


    block3 = Block()
    block3.set_stride("1x1")
    cell31 = Cell(input1 = ConvolutionInput((5,5),(1,1),120,"valid", "tanh"))
    block3.append_cell(cell31)

    block4 = Block()
    block4.set_stride("1x1")
    cell41 = Cell(input1 = DenseInput(84, "tanh"))
    block4.append_cell(cell41)

    blocks.extend([block1, block2,block22, block3, block4])

    return blocks
