
import copy , sys, json
sys.path.append("../..")
import os, pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import random, time
import getopt
from tensorflow_generator import TensorflowGenerator
from model.keras_model import KerasFeatureModel
from model.node import Node
from model.mutation.mutable_parameters import MutableParameters
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def _get_layer_id(layer_name):
    return layer_name[0:layer_name.find("-n-")]


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

def run(experiment_path=".", mutation_config_path="./light_config.json"):
    os.makedirs('{}/output/'.format(experiment_path), exist_ok=True)

    config = MutableParameters.load_config(mutation_config_path)

    if not config or not config.get("evolution_parameters"):
        return

    config = config.get("evolution_parameters")
    model_name = config.get("model_name")
    training_epochs= config.get("training_epochs")
    dataset = config.get("dataset")

    attacks = config.get("attacks")
    mutation_ratio = config.get("mutation_ratio")
    nb_mutations = config.get("nb_mutations")
    nb_mutants = config.get("nb_mutants")
    robustness_set_size = config.get("robustness_set_size")

    random_seed = config.get("random_seed",None)
    if random_seed:
        random.seed(random_seed)
        np.random.seed(random_seed)

    tensorflow_gen = TensorflowGenerator(model_name, training_epochs, dataset)
    original_model = tensorflow_gen.model.model

    node_name = Node.node_list[1]
    node_name = _get_layer_id(node_name)
    all_layers = [(e.name,e) for e in original_model.layers]
    layers_to_copy = [(_get_layer_id(e),v.get_weights()) for (e,v) in all_layers if e.find(node_name) == -1]

    mutants = generate_mutants(tensorflow_gen.model, node_name, nb_mutants=nb_mutants, nb_mutations=nb_mutations, mutation_ratio=mutation_ratio)


    #pickle.dump(mutants, open('{}/output/step1_mutants.pickle'.format(experiment_path), "wb"))
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
            history = {"mutant":"{}_{}".format(i,mutant.name),"train_acc":history.history['acc'], "test_acc":history.history['val_acc'], **robustness, "mutations":mutant.mutation_history, **time}
            print("mutant {}".format(i), history)
            histories.append(history)
            with open('{}/output/step1_metrics.json'.format(experiment_path), 'w') as file:
                json.dump(histories,file)


def main(argv):
    experiment_path = './{}'.format(int(time.time()))
    mutation_config_path = "./light_config.json"


    try:
        opts, args = getopt.getopt(argv, "hx:c:", [
            "xp_path=", "config_path="])
    except getopt.GetoptError:
        pass
    print("arguments {}".format(opts))
    for opt, arg in opts:
        if opt == '-h':
            print(
                'step1.py -x <experience_path> -c <config_path>')
            sys.exit()
        elif opt in ("-x", "--xp"):
            experiment_path = arg
        elif opt in ("-c", "--config_path"):
            mutation_config_path = arg

    run(experiment_path,mutation_config_path)

if __name__ == "__main__":
    main(sys.argv[1:])

