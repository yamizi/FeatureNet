import copy, sys
sys.path.append("../..")

from experiments.local_robustness.step1 import run
from model.mutation.mutable_parameters import MutableParameters


variants = [
    {"mutable_node": 8,"experiment_path": "local/cifar10/n8_noAugment"},
    {"mutable_node": 23,"experiment_path": "local/cifar10/n23_noAugment"},
    {"mutable_node": 30,"experiment_path": "local/cifar10/n2_noAugment"}
]

if __name__ == "__main__":
    mutation_config_path = "./configurations/light_config_node1.json"
    config_original = MutableParameters.load_config(mutation_config_path)

    for i, variant in enumerate(variants):
        print("building varian {}".format(i))
        config = copy.copy(config_original)
        config["evolution_parameters"].update(variant)
        run("{}_{}".format(i,mutation_config_path), config)




