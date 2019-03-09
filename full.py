
from tensorflow_generator import TensorflowGenerator
from products_tree import ProductSet
import multiprocessing
import json




def run_tensorflow(product, url, index,datasets=[], epochs=12):
        for dataset in datasets:
                tensorflow = TensorflowGenerator(product,epochs, dataset)
                f2 = open("report_600_{0}_{1}.txt".format(url,dataset),"a")
                f2.write("\r\n{0}: {1} {2} {3} {4} {5}".format(index, tensorflow.accuracy, tensorflow.stop_training, tensorflow.training_time, tensorflow.params, tensorflow.flops))
                f2.close()

def main(target, min_index=0, max_index=0, filter_indices=[], datasets=None,epochs=12):
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

                        run_tensorflow(product, target, index, datasets, epochs)
                        #p = multiprocessing.Process(target=run_tensorflow, args=(product,))
                        #p.start()
                        #p.join()

                if max_index!= 0 and index ==max_index:
                        break

if __name__ == "__main__":
    # execute only if run as a script
    main("1000Products", datasets=["cifar"], filter_indices=[],epochs=600)
       

        
       
        
        

        

        


        



