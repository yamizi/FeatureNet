from pledge_evolution import PledgeEvolution
from tensorflow_generator import TensorflowGenerator
from model.keras_model import KerasFeatureVector
from products_tree import ProductSet, ProductSetError

#PledgeEvolution.end2end("../products/run12","1x5x100".split("x"))
from numpy.random import choice
import os, sys
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

TensorflowGenerator.model_graph = "./model_graphs/lenet_mutants/original"
tensorflow = TensorflowGenerator("lenet5",300, "cifar",no_train=True)

ref_blocks = [tensorflow.model.blocks]
for i in range(10):

        tensorflow.model.blocks = ref_blocks[choice(len(ref_blocks))]
        tensorflow.model.mutate()
        ref_blocks.append(tensorflow.model.blocks)
        tensorflow.model.build(TensorflowGenerator.input_shape, 10)
        TensorflowGenerator.model_graph = "./model_graphs/lenet_mutants/m_{}".format(i)
        tensorflow.print(include_summary=True, invalid_params=False)
        

sys.exit()

pdt_file = "./pledge_product/1000Products_splc.pdt"
pdt_file ="../products/run18/1000products.pdt"

initial_product_set, last_population = PledgeEvolution.extract_leaves(pdt_file)#("../products/run11/cifar/e0_m0_b0.pdt")
for i, (product, original) in enumerate(initial_product_set.format_products()):
    TensorflowGenerator.model_graph = "./model_graphs/model18_{}".format(i)
    tensorflow = TensorflowGenerator(product, product_features=original,features_label=initial_product_set.features, depth=5, no_train=True)

    vector = tensorflow.model.to_kerasvector()
    if i >= 0:
        break