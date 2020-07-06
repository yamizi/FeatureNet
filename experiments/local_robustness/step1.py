
import copy , sys
sys.path.append("../..")

from tensorflow_generator import TensorflowGenerator
from model.keras_model import KerasFeatureModel
from model.node import Node
from model.mutation.mutable_base import MutableBase, MutationStrategies, SelectionStrategies

def generate_mutants(fm_model, node_name,nb_mutants=1, nb_mutations=1,mutation_ratio = 1):
    mutants = []
    original_blocks = fm_model.dump_blocks()
    for i in range(nb_mutants):
        blocks = copy.deepcopy(original_blocks)
        mutant = KerasFeatureModel.parse_blocks({"blocks": blocks["blocks"]})
        for j in nb_mutations:
            mutant.mutate(mutation_ratio)



def run():
    model_name, training_epochs, dataset = "keras", 1, "cifar"
    attacks = ["cw", "pgd"]
    mutation_ratio = 0.1
    nb_mutations = 1
    tensorflow_gen = TensorflowGenerator(model_name, training_epochs, dataset, no_train=True)
    model = tensorflow_gen.model.model


def run2():
    model_name, training_epochs, dataset = "keras", 1, "cifar"
    attacks = ["cw", "pgd"]
    mutation_ratio = 0.1
    nb_mutations = 1
    tensorflow_gen = TensorflowGenerator(model_name, training_epochs, dataset, data_augmentation=False)
    #TensorflowGenerator.eval_robustness(tensorflow_gen.model, attacks,100)

    blocks1 = tensorflow_gen.model.dump_blocks()
    node_mapping1 = list(Node.node_mapping.keys())
    layer_mapping1 = list(Node.layer_mapping.keys())
    node_list1 = Node.node_list.copy()

    blocks = copy.deepcopy(blocks1)
    for b in blocks1["blocks"]:
        for c in b.cells:
            print(c.uniqid)

    mutant = KerasFeatureModel.parse_blocks({"blocks": blocks["blocks"]})

    all_layers = [e.name for e in tensorflow_gen.model.model.layers]
    for l in layer_mapping1:
        if l.find("Conv") > -1:
            tensorflow_gen.model.model.get_layer(l).trainable = False


    for j in range(nb_mutations):
        #mutant.mutate(mutation_ratio)
        tensorflow_gen = TensorflowGenerator(mutant, training_epochs, dataset, data_augmentation=False)
        TensorflowGenerator.eval_robustness(tensorflow_gen.model, attacks, 100)

        node_mapping = list(Node.node_mapping.keys())
        node_mapping2 =  [e for e in node_mapping if e not in node_mapping1]
        layer_mapping = list(Node.layer_mapping.keys())
        layer_mapping2 = [e for e in layer_mapping if e not in layer_mapping1]
        node_list2 = Node.node_list #[e for e in Node.node_list if e not in node_list1]

        print(node_mapping1)
        print(node_mapping2)
        print(node_list1)
        print(node_list2)



        TensorflowGenerator.train(tensorflow_gen.model, training_epochs, dataset, False)
        TensorflowGenerator.eval_robustness(tensorflow_gen.model, attacks, 100)

        #print(layer_mapping1)
        #print(layer_mapping2)

#// Freeze the first layer.
#layer0.trainable = false;

        blocks2 = tensorflow_gen.model.dump_blocks()



run()
