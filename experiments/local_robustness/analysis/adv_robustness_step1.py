import sys, json
sys.path.append("../..")

from tensorflow_generator import TensorflowGenerator
import tensorflow as tf
from tensorflow import keras

def run(model_path="node1/1597067774/keras411c36b4-d.h5", robustness_set_size=1000):
    sess = tf.Session()
    graph = tf.get_default_graph()

    path = "../output/models/{}".format(model_path)
    dataset = "cifar"
    norm = 2

    with graph.as_default():
        keras.backend.set_session(sess)

        TensorflowGenerator.init_dataset(dataset)
        reconstructed_model = keras.models.load_model(path)
        y_pred = reconstructed_model.predict(TensorflowGenerator.X_robustness)
        pgd_score = TensorflowGenerator.eval_attack_robustness(reconstructed_model, "pgd", norm, robustness_set_size, session=sess)
        #print(pgd_score)

run("node8/aa5ebc02-d.h5")


