import sys
sys.path.append("..")

from tensorflow_generator import TensorflowGenerator
from Products_tree import ProductSet
import json




def run_tensorflow(product, url, index,datasets=[], epochs=12, depth=1, data_augmentation=False):
        for dataset in datasets:
                logpath = "report_top_{2}epochs_{3}depth_{0}_{1}.txt".format(url,dataset, epochs, depth)
                tensorflow = TensorflowGenerator(product,epochs, dataset, depth=depth, data_augmentation=data_augmentation)
                f2 = open(logpath,"a")

                history = "{accuracy}|{validation_accuracy}".format(accuracy="#".join(map(str, tensorflow.history[0])), validation_accuracy="#".join(map(str, tensorflow.history[1])))
                f2.write("\r\n{0}: {1} {2} {3} {4} {5} {6}".format(index, tensorflow.accuracy, tensorflow.stop_training, tensorflow.training_time, tensorflow.params, tensorflow.flops, history))
                f2.close()

def main(target, min_index=0, max_index=0, filter_indices=[], datasets=None,epochs=12, depth=1, data_augmentation=False):
        baseurl = "./"
        productSet = ProductSet(baseurl+target+".pdt")

        if not datasets:
                datasets = ["mnist"]

        for index,product in enumerate(productSet.format_products()):
                print("product {0}".format(index))

                if index >= min_index and (len(filter_indices)==0 or index in filter_indices):
                        f = open("{0}products/{1}_{2}.json".format(baseurl, target, index), "w")
                        str_ = json.dumps(product)
                        f.write(str_)
                        f.close()
                        run_tensorflow(product, target, index, datasets, epochs, depth, data_augmentation=data_augmentation)


                if max_index!= 0 and index ==max_index:
                        break

if __name__ == "__main__":
    # execute only if run as a script
    TensorflowGenerator.training_loss = 'binary_crossentropy'
    tensorflow = TensorflowGenerator("keras_lstm",15, "imdb_sentiment", depth=1, data_augmentation=False, name="lstm",clear_memory=False )
