
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

def save_weights(layers, save_path, target_model=None):

    if target_model is not None:
        map_layers_target = [(_get_layer_id(e.name), e) for e in target_model.layers]
        names = list(layers.keys())

        for (name, layer) in map_layers_target:
            if name in names:
                try:
                    np.save("{}_{}".format(save_path, name), layer.get_weights())
                except Exception as err:
                    print("error {} saving weight of layer {}".format(err, name))

    else:
        for e,v in layers.items():
            try:
                np.save("{}_{}".format(save_path,e), v)
            except Exception as err:
                print("error {} saving weight of layer {}".format(err, e))

def run(mutation_config_path="./light_config.json"):

    config = MutableParameters.load_config(mutation_config_path)

    if not config or not config.get("evolution_parameters"):
        return

    config = config.get("evolution_parameters")
    model_name = config.get("model_name")
    training_epochs= config.get("training_epochs")
    dataset = config.get("dataset")

    data_augmentation = config.get("data_augmentation", False)
    batch_size = config.get("batch_size", 128)

    attacks = config.get("attacks")
    mutation_ratio = config.get("mutation_ratio")
    nb_mutations = config.get("nb_mutations")
    nb_mutants = config.get("nb_mutants")
    robustness_set_size = config.get("robustness_set_size")
    mutable_node = config.get("mutable_node", 1)

    experiment_path = "{}/{}".format(config.get("experiment_path",'.'), int(time.time()))
    model_path = 'output/models/{}'.format(experiment_path)
    weights_path = 'output/weight/{}'.format(experiment_path)
    metrics_path = 'output/metrics/{}'.format(experiment_path)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(weights_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)

    random_seed = config.get("random_seed",None)
    if random_seed:
        random.seed(random_seed)
        np.random.seed(random_seed)

    tensorflow_gen = TensorflowGenerator(model_name, training_epochs, batch_size=batch_size,
                                                                                   dataset=dataset, data_augmentation=data_augmentation,save_path="{}/{}".format(model_path,model_name))
    original_model = tensorflow_gen.model.model

    node_name = Node.node_list[mutable_node]
    node_name = _get_layer_id(node_name)
    all_layers = [(e.name,e) for e in original_model.layers]
    layers_to_copy = [(_get_layer_id(e),v.get_weights()) for (e,v) in all_layers if e.find(node_name) == -1]

    layer_save_path = "{}/{}".format(weights_path, model_name)
    save_weights(dict(all_layers), layer_save_path)

    mutants = generate_mutants(tensorflow_gen.model, node_name, nb_mutants=nb_mutants, nb_mutations=nb_mutations, mutation_ratio=mutation_ratio)


    #pickle.dump(mutants, open('{}/output/step1_mutants.pickle'.format(experiment_path), "wb"))
    histories = []
    for i, mutant in enumerate(mutants):
        tensorflow_gen = TensorflowGenerator(mutant, training_epochs, dataset, no_train=True, clear_memory=True)

        if tensorflow_gen.valid:
            layer_save_path = "{}/{}".format(weights_path, mutant.name)
            save_weights(dict(all_layers), layer_save_path,tensorflow_gen.model.model)
            #copy_weights(tensorflow_gen.model.model, dict(layers_to_copy), True)
            history, training_time, score, keras_model = TensorflowGenerator.train(tensorflow_gen.model, training_epochs, batch_size=batch_size,
                                                                                   dataset=dataset, data_augmentation=data_augmentation,save_path="{}/{}".format(model_path,mutant.name))

            robustness_time = TensorflowGenerator.eval_robustness(tensorflow_gen.model, attacks, robustness_set_size)
            robustness = {"robustness_score":tensorflow_gen.model.robustness_score}
            infos = {"training_time":training_time, "robustness_time":robustness_time, "config_path":mutation_config_path}
            history = {"mutant":"{}_{}".format(i,mutant.name),"train_acc":history.history['acc'], "test_acc":history.history['val_acc'], **robustness, "mutations":mutant.mutation_history, **infos}
            print("mutant {}".format(i), history)
            histories.append(history)
            with open('{}/step1_metrics.json'.format(metrics_path), 'w') as file:
                json.dump(histories,file)


def main(argv):
    mutation_config_path = "./configurations/light_config_node1.json"

    try:
        opts, args = getopt.getopt(argv, "hc:", ["config_path="])
    except getopt.GetoptError:
        pass
    print("arguments {}".format(opts))
    for opt, arg in opts:
        if opt == '-h':
            print(
                'step1.py -c <config_path>')
            sys.exit()
        elif opt in ("-c", "--config_path"):
            mutation_config_path = arg

    TensorflowGenerator.model_graph_export = False
    run(mutation_config_path)

if __name__ == "__main__":
    main(sys.argv[1:])

