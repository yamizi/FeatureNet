
from tensorflow_generator import TensorflowGenerator
from products_tree import ProductSet
import json
import sys, getopt, os



def run_tensorflow(product, url, index,datasets=[], epochs=12, depth=1, data_augmentation=False):
        for dataset in datasets:
                logpath = "report_all_{2}epochs_{3}depth_{0}_{1}.txt".format(url,dataset, epochs, depth)
                tensorflow = TensorflowGenerator(product,epochs, dataset, depth=depth, data_augmentation=data_augmentation)
                f2 = open(logpath,"a")

                history = "{accuracy}|{validation_accuracy}".format(accuracy="#".join(map(str, tensorflow.history[0])), validation_accuracy="#".join(map(str, tensorflow.history[1])))
                f2.write("\r\n{0}: {1} {2} {3} {4} {5} {6}".format(index, tensorflow.accuracy, tensorflow.stop_training, tensorflow.training_time, tensorflow.params, tensorflow.flops, history))
                f2.close()

def main(target, min_index=0, max_index=0, filter_indices=[], datasets=None,epochs=12, depth=1, data_augmentation=False, output_folder=""):
        productSet = ProductSet(target+".pdt"
        )
        if not output_folder:
                output_folder = "./products/"

        if not datasets:
                datasets = ["mnist"]

        for index,product in enumerate(productSet.format_products()):
                print("product {0}".format(index))

                if index >= min_index and (len(filter_indices)==0 or index in filter_indices):
                        f = open("{0}{1}_{2}.json".format(output_folder, target, index), "w")
                        str_ = json.dumps(product)
                        f.write(str_)
                        f.close()
                        run_tensorflow(product, target, index, datasets, epochs, depth, data_augmentation=data_augmentation)

                if max_index!= 0 and index ==max_index:
                        break



def init(argv):
    input_file = '1000Products'
    depth=1
    datasets = ["cifar"]
    training_epochs = 600
    output_folder = "./products/"

    try:
        opts, args = getopt.getopt(argv, "hi:n:d:t:", [
                                   "ifile=", "depth=","datasets=","training_epoch="])
    except getopt.GetoptError:
        pass
    print("arguments {}".format(opts))
    for opt, arg in opts:
        
        if opt == '-h':
            print(
                'full.py -i <input_file> -n <depth> -d <datasets> -t <training_epoch>')
            sys.exit()
        elif opt in ("-n", "--depth"):
            depth = int(arg)
        elif opt in ("-d", "--dataset"):
            datasets = arg.split(";")
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-t", "--training_epoch"):
            training_epochs = int(arg)

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
        
    main(input_file, datasets=datasets, epochs=training_epochs, depth=depth, output_folder=output_folder)

if __name__ == "__main__":
    init(sys.argv[1:])

       

        
       
        
        

        

        


        



