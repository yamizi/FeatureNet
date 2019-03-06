
from tensorflow_generator import TensorflowGenerator
from products_tree import ProductSet
import json

baseurl = "./"
url = "1000Products"
productSet = ProductSet(baseurl+url+".pdt")

products = []
min_index = 0



for index,product in enumerate(productSet.format_products()):
    print("product {0}".format(index))

    if index >= min_index:
        f = open("{0}products/{1}_{2}.json".format(baseurl, url, index), "w")
        str_ = json.dumps(product)
        f.write(str_)
        f.close()

        datasets = ["mnist", "cifar"]
        for dataset in datasets:
                tensorflow = TensorflowGenerator(product,12, dataset)
                f2 = open("report_{0}_{1}.txt".format(url,dataset),"a")
                f2.write("\r\n{0}: {1} {2} {3} {4}".format(index, tensorflow.accuracy, tensorflow.stop_training, tensorflow.training_time, tensorflow.params))
                f2.close()
                products.append(tensorflow.model)
        

        

        


        




