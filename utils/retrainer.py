import sys
sys.path.append("./")
from tensorflow_generator import TensorflowGenerator
import os
import keras

os.environ["CUDA_VISIBLE_DEVICES"]="-1"   

def run(model_path, epochs=100, dataset="cifar", data_augmentation = True, batch_size=64, save_path="./saved_models"):

    TensorflowGenerator.init_dataset(dataset)
    if os.path.isfile(model_path):
        print("Loading existing model {}".format(model_path))
        model = keras.models.load_model(model_path)
        
        TensorflowGenerator.train(model, epochs, batch_size, dataset, data_augmentation, save_path=save_path)

run("C:/Users/salah.ghamizi/Documents/PhD/Code/products/2metrics/cifar#lenet5/ee10_te12_mr0.1_sr0.2/e3_57a6b089-a.h5")
            