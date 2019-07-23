from .input import Input, ZerosInput, PoolingInput, ConvolutionInput, DenseInput, IdentityInput
from .output import Output, OutCell, OutBlock, Out
from .operation import Operation, Combination, Sum, Flat, Drop
from .block import Block
from .cell import Cell

def standard_blocks():
    blocks = []

    block1 = Block()
    cell11 = Cell(input1 = ConvolutionInput((3,3),(1,1),32,"same", "relu"))
    block1.append_cell(cell11)
    cell12 = Cell(input1 = ConvolutionInput((3,3),(1,1),32,"same", "relu"))
    block1.append_cell(cell12)
    cell13 = Cell(input1 = PoolingInput((2,2),(1,1),"max", "valid"), operation1=Drop(0.25))
    block1.append_cell(cell13)


    block2 = Block()
    cell21 = Cell(input1 = ConvolutionInput((3,3),(1,1),64,"same", "relu"))
    block2.append_cell(cell21)
    cell22 = Cell(input1 = ConvolutionInput((3,3),(1,1),64,"same", "relu"))
    block2.append_cell(cell22)
    cell13 = Cell(input1 = PoolingInput((2,2),(1,1),"max", "valid"), operation1=Drop(0.25))
    block1.append_cell(cell13)

    block4 = Block()
    cell41 = Cell(input1 = DenseInput(512, "relu"), operation1=Drop(0.25))
    block4.append_cell(cell41)

    blocks.extend([block1, block2,block4])

    return blocks