
from tensorflow_generator import TensorflowGenerator
from model.keras_model import KerasFeatureVector, KerasFeatureModel
from products_tree import ProductSet, ProductSetError
import json, sys, getopt
import numpy as np

from pledge_evolution import PledgeEvolution


def train_model_from_product(pledge_output_file,index,export_file="", training_epochs=5, batch_size=64, dataset="mnist"):
    initial_product_set = ProductSet(pledge_output_file, binary_products=True)

    product, original_product = initial_product_set.format_product(prd_index=index)
    tensorflow = TensorflowGenerator(product, training_epochs, dataset, product_features=original_product, features_label=initial_product_set.features, batch_size=batch_size)
    vector_pdt = tensorflow.model.to_kerasvector()

    #initial_product_set.products = [vector_pdt.features]
    #product, original_product = initial_product_set.format_product(prd_index=index, sort_features=False)
    parsed_product = initial_product_set.format_product(original_product=vector_pdt.features)

    if not parsed_product:
        return None
    product, original_product = parsed_product
    tensorflow = TensorflowGenerator(product, training_epochs, dataset, product_features=original_product, features_label=initial_product_set.features, batch_size=batch_size)

    f1 = open(export_file, 'w')
    f1.write(json.dumps([vector_pdt.to_vector()]))
    f1.close()

def train_model_from_json(pledge_output_file,products_file,index=None,export_file="", training_epochs=5, batch_size=64, dataset="mnist"):

    initial_product_set = ProductSet(pledge_output_file, binary_products=True)
    
    f1 = open(products_file, 'r')
    vects = json.loads(f1.read())
    products = [KerasFeatureVector.from_vector(vect) for vect in vects]
    vectors_export = []
    

    if index is not None: 
        if len(index) ==0:
            products = [products[int(index)]]
        else:
            products = products[max(0,int(index[0])):min(len(products),int(index[1]))]

    
    for i,keras_product in enumerate(products):
        initial_product_set.binary_products = True
        product, original_product = initial_product_set.format_product(original_product=keras_product.features)
        
        tensorflow = TensorflowGenerator(product, training_epochs, dataset, product_features=original_product, features_label=initial_product_set.features, batch_size=batch_size)
        print("original accuracy {} new accuracy {}".format(keras_product.accuracy, tensorflow.model.accuracy))
        vector_pdt = tensorflow.model.to_kerasvector()
        vectors_export.append(vector_pdt.to_vector())

    if export_file:
        f1 = open(export_file, 'w')
        f1.write(json.dumps(vectors_export))
        f1.close()


#train_model_from_product('../products/run10/mnist/e0_m0_b0.pdt',0,"../products/run10/mnist/e0_m0_b0_2.json",1)
#train_model_from_json('../products/run10/mnist/e0_m0_b0.pdt',"../products/run10/mnist/e0_m0_b0_2.json",0,"../products/run10/mnist/e0_m0_b0_3.json",1)
#train_model_from_json('../products/run10/mnist/e0_m0_b0.pdt',"../products/run10/mnist/e0_m0_b0_3.json",0,1)

if __name__ == "__main__":
    argv = sys.argv[1:]

    pledge_product = json_product = export_file = ""
    index_product = None
    dataset = "cifar"
    batch_size = 0
    training_epochs = 300

    try:
        opts, args = getopt.getopt(argv, "hp:j:i:e:t:b:d:", [
                                   "pledge=","json=",  "index=","export=", "traing_epoch=", "batch_size=", "dataset="])
    except getopt.GetoptError:
        pass
    print("arguments {}".format(opts))
    
    for opt, arg in opts:
        
        if opt == '-h':
            print(
                'pledge_trainer.py -p <pledge_product> -j <json_product> -i <index_product> -e <export_file> -t <training_epochs> -b <batch_size> -t <dataset>')
            sys.exit()
        elif opt in ("-p", "--pledge_product"):
            pledge_product = arg
        elif opt in ("-j", "--json_product"):
            json_product = arg
        elif opt in ("-e", "--export_file"):
            export_file = arg
        elif opt in ("-i", "--index"):
            index_product = arg.split("-")
        elif opt in ("-t", "--training_epochs"):
            training_epochs = int(arg)
        elif opt in ("-b", "--batch_size"):
            batch_size = int(arg)
        elif opt in ("-d", "--dataset"):
            dataset = arg

    if json_product:
        train_model_from_json(pledge_product,json_product,index_product,export_file, training_epochs, batch_size, dataset)
    else:
        train_model_from_product(pledge_product,index_product,export_file, training_epochs, batch_size, dataset)