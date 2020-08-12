import sys, json
sys.path.append("../..")

import tensorflow as tf
from tensorflow import keras

def run(models_path="node8\1597081929"):

    path = "../output/models/{}".format(models_path)
    reconstructed_model = keras.models.load_model(path)

