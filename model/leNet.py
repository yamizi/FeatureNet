from .input import Input, ZerosInput, PoolingInput, ConvolutionInput, DenseInput, IdentityInput
from .output import Output, OutCell, OutBlock, Out
from .operation import Operation, Combination, Sum, Flat
from .block import Block
from .cell import Cell

def lenet5_blocks():
    blocks = []

    block1 = Block()
    cell11 = Cell(input1 = ConvolutionInput((5,5),(1,1),6,"same", "tanh"))
    block1.append_cell(cell11)
    cell12 = Cell(input1 = PoolingInput((2,2),(1,1),"average", "valid"))
    block1.append_cell(cell12)

    block2 = Block()
    cell21 = Cell(input1 = ConvolutionInput((5,5),(1,1),16,"same", "tanh"))
    block2.append_cell(cell21)
    cell22 = Cell(input1 = PoolingInput((2,2),(2,2),"average", "valid"))
    block2.append_cell(cell22)


    block3 = Block()
    cell31 = Cell(input1 = ConvolutionInput((5,5),(1,1),120,"valid", "tanh"))
    block3.append_cell(cell31)

    block4 = Block()
    cell41 = Cell(input1 = DenseInput(84, "tanh"))
    block4.append_cell(cell41)

    blocks.extend([block1, block2, block3, block4])

    return blocks