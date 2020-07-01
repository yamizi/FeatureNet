from .input import EmbeddingInput, DenseInput, LSTMInput, PoolingInput, ConvolutionInput, IdentityInput
from .operation import  Drop, Active
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


def lstm_blocks(max_features = 20000,lstm_units = 128):
    blocks = []

    block1 = Block()
    cell11 = Cell(input1 = EmbeddingInput(max_features))
    block1.append_cell(cell11)

    block2 = Block()
    cell21 = Cell(input1 = LSTMInput(lstm_units))
    block2.append_cell(cell21)

    block3 = Block()
    cell31 = Cell(input1 = DenseInput(1, "sigmoid"))
    block3.append_cell(cell31)

    blocks.extend([block1, block2, block3])

    return blocks


def cnn1d_blocks(input_dims = 20000,maxlen = 400,output_dims = 50,filters = 250,hidden_dims = 250):
    blocks = []

    block1 = Block()

    cell11 = Cell(input1 = EmbeddingInput(input_dims=input_dims,_output_dim=output_dims, _input_length=maxlen))
    block1.append_cell(cell11)

    block2 = Block()
    block2.set_stride("1x1")
    cell21 = Cell(input1 = ConvolutionInput(_features=filters,_activation="none"))
    block2.append_cell(cell21)
    cell22 = Cell(input1=PoolingInput(_features=2, _stride="none"))
    block2.append_cell(cell22)
    cell23 = Cell(input1=DenseInput(hidden_dims, "none"), operation1=Drop(20))
    block2.append_cell(cell23)
    cell24 = Cell(input1=IdentityInput(), operation1=Active("relu"))
    block2.append_cell(cell24)

    block3 = Block()
    block3.set_stride("1x1")
    cell31 = Cell(input1=DenseInput(1, "sigmoid"))
    block3.append_cell(cell31)

    blocks.extend([block1, block2, block3])
    return blocks
