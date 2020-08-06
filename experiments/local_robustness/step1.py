
import copy , sys, json
sys.path.append("../..")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from tensorflow_generator import TensorflowGenerator
from model.keras_model import KerasFeatureModel
from model.node import Node
from model.mutation.mutable_base import MutableBase, MutationStrategies, SelectionStrategies
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def _get_layer_id(layer_name):
    return layer_name[0:layer_name.find("-c")]


def generate_mutants(fm_model, node_name,nb_mutants=1, nb_mutations=1,mutation_ratio = 1):
    mutants = []
    original_blocks = fm_model.dump_blocks()
    node_mapping1 = list(Node.node_mapping.keys())

    for i in range(nb_mutants):
        blocks = copy.deepcopy(original_blocks)
        mutant = KerasFeatureModel.parse_blocks({"blocks": blocks["blocks"]})

        node_mapping2 = list(Node.node_mapping.keys())
        node_mapping = [e for e in node_mapping2 if e not in node_mapping1]
        node_relevant = [e for e in node_mapping if e.find(node_name)>-1]
        node_mapping1 = node_mapping2
        node =Node.node_mapping[node_relevant[0]]
        print("new mutant",node_name, node.name)

        for j in range(nb_mutations):
            node.mutate(mutation_ratio)

        mutant.mutation_history.append(node.mutations)
        mutants.append(mutant)

    return mutants

def copy_weights(target_model, src_layers=None, freeze_layers=False):

    map_layers_target = [(_get_layer_id(e.name),e) for e in target_model.layers]

    names = list(src_layers.keys())

    for (name,layer) in map_layers_target:
        if name in names:

            try:
                layer.set_weights(src_layers[name])
                layer.trainable = not freeze_layers
            except:
                pass
    TensorflowGenerator.compile(keras_model=target_model)

    return target_model

def run(experiment_path="./experiments/local_robustness"):
    experiment_path = "."
    model_name, training_epochs, dataset = "keras", 10, "cifar"
    attacks = ["pgd"]
    mutation_ratio = 0.1
    nb_mutations = 1
    nb_mutants = 10
    robustness_set_size = 1000

    tensorflow_gen = TensorflowGenerator(model_name, training_epochs, dataset)
    original_model = tensorflow_gen.model.model

    node_name = Node.node_list[1]
    node_name = _get_layer_id(node_name)
    all_layers = [(e.name,e) for e in original_model.layers]
    layers_to_copy = [(_get_layer_id(e),v.get_weights()) for (e,v) in all_layers if e.find(node_name) == -1]

    mutants = generate_mutants(tensorflow_gen.model, node_name, nb_mutants=nb_mutants, nb_mutations=nb_mutations, mutation_ratio=mutation_ratio)

    histories = []
    for i, mutant in enumerate(mutants):
        tensorflow_gen = TensorflowGenerator(mutant, training_epochs, dataset, no_train=True, clear_memory=True)

        if tensorflow_gen.valid:
            copy_weights(tensorflow_gen.model.model, dict(layers_to_copy), True)
            history, training_time, score, keras_model = TensorflowGenerator.train(tensorflow_gen.model, training_epochs, batch_size=128,
                                                                                   dataset=dataset, data_augmentation=False,save_path=None)

            robustness_time = TensorflowGenerator.eval_robustness(tensorflow_gen.model, attacks, robustness_set_size)
            robustness = {"robustness_score":tensorflow_gen.model.robustness_score}
            time = {"training_time":training_time, "robustness_time":robustness_time}
            history = {"train_acc":history.history['acc'], "test_acc":history.history['val_acc'], **robustness, "mutations":mutant.mutation_history, **time}
            print("mutant {}".format(i), history)
            histories.append(history)

    print(histories)
    with open('{}/step1.json'.format(experiment_path), 'w') as outfile:
        json.dump(histories,outfile)

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
