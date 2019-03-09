
from tensorflow_generator import TensorflowGenerator
from products_tree import ProductSet
import multiprocessing
import json

baseurl = "./"
url = "100Products_lenet5_constrained"
productSet = ProductSet(baseurl+url+".pdt")

min_index = 0


def run_tensorflow(product):
        datasets = ["mnist"]
        for dataset in datasets:
                tensorflow = TensorflowGenerator(product,12, dataset)
                f2 = open("reports/report_{0}_{1}.txt".format(url,dataset),"a")
                f2.write("\r\n{0}: {1} {2} {3} {4} {5}".format(index, tensorflow.accuracy, tensorflow.stop_training, tensorflow.training_time, tensorflow.params, tensorflow.flops))
                f2.close()

for index,product in enumerate(productSet.format_products()):
    print("product {0}".format(index))

    if index >= min_index:
        f = open("{0}products/{1}_{2}.json".format(baseurl, url, index), "w")
        str_ = json.dumps(product)
        f.write(str_)
        f.close()

        run_tensorflow(product)
        #p = multiprocessing.Process(target=run_tensorflow, args=(product,))
        #p.start()
        #p.join()

       

        
       
        
        

        

        


        



