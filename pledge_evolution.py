import subprocess
import sys, getopt, json, os
from tensorflow_generator import TensorflowGenerator
from model.keras_model import KerasFeatureVector
from products_tree import ProductSet
import random
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import tensorflow
import gc


pledge_path = '../products/PLEDGE.jar'
base_path = '../products/run6/'
training_epochs = 50
evolution_epochs = 20

def run_pledge(input_file,nb_base_products, output_file):
   params = ['java', '-jar', pledge_path,'generate_products']
   params = params+['-fm',input_file,'-nbProds',str(nb_base_products),'-o',output_file]
   print("running pledge on {}".format(input_file))
   subprocess.check_call(params)
   return 

def select(last_population,survival_count):
   labels = last_population.get("filtered_features_label")
   products = last_population.get("products")
   products = sorted(products, key=lambda x: x.accuracy, reverse=True)[:survival_count]

   products_labels = [[labels[k] for k in labels.keys() if prd.features[int(k)] ==1] for prd in products]
   return products, products_labels

def mutate(mutant_id, mutant_labels, initial_feature_model, nb_products, constraints_ratio=0.5):

   mutant_labels = [mutant for mutant in mutant_labels if random.random()<constraints_ratio ]

   import xml.etree.ElementTree as ET
   tree = ET.parse(initial_feature_model)
   root = tree.getroot()
   constraints = root.getchildren()[1]
   constraints_raw = constraints.text.split("\n")
   nb_constraints = len(constraints_raw)
   constraints_raw = constraints_raw[:-1]
   for i, lbl in enumerate(mutant_labels):
      constraints_raw.append("C{}:~Architecture  or  ~{}".format(nb_constraints+i-1, lbl)) 
   
   constraints.text = "\n".join(constraints_raw)
   dst = "{}_{}.xml".format(initial_feature_model[:-4],mutant_id)
   tree.write(dst, encoding="UTF-8", xml_declaration=True)

   output_file = "{}.{}".format(dst[:-4],"pdt")
   run_pledge(dst,nb_products, output_file)

   # possibility to generate multiple products each containing a different set of constraints 
   return output_file


def extract_leaves(output_file):
   initial_product_set = ProductSet(output_file)
   filtered_features_label = ProductSet.filter_leaves(initial_product_set.features)
   last_population = {"filtered_features_label":[], "products":[]}
   last_population["filtered_features_label"]=filtered_features_label

   return initial_product_set, last_population


def reset_keras(classifier):
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del classifier  
    except:
        pass

    print(gc.collect()) # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = tensorflow.ConfigProto()
    #if hasattr(config, "gpu_options"):
    #  config.gpu_options.per_process_gpu_memory_fraction = 1
    #  config.gpu_options.visible_device_list = "0"
    set_session(tensorflow.Session(config=config))

def train_products(initial_product_set, output_file, dataset):
   print("training products in {}".format(output_file))
   last_population = []
   for index, (product, original_product) in enumerate(initial_product_set.format_products()):
      print("### training product {}".format(index))
      tensorflow = TensorflowGenerator(product,training_epochs, dataset, product_features=original_product, features_label=initial_product_set.features, no_train=False)    
      last_population.append(tensorflow.model.to_vector())
      reset_keras(tensorflow)
        
   return last_population

def run(input_file="",output_file="", last_pdts_path="", nb_base_products=50, dataset="cifar"):

    
   survival_count = int(0.2*nb_base_products)
   nb_product_perparent = 5
   last_evolution_epoch = 0

   if not input_file:
      input_file = '{}main_5_5.xml'.format(base_path)  
      #input_file = '{}main_1block_10_cells.xml'.format(base_path)  

   if not output_file:
      output_file = '{}main_5_5_{}products.pdt'.format(base_path,nb_base_products)  
      #output_file = '{}main_1blocks_10cells_{}products.pdt'.format(base_path,nb_base_products)  

   if not last_pdts_path:
      p = output_file.find("products")
      if p > -1:
         last_pdts_path = "{}_{}.json".format(output_file[0:p],dataset )
      else:
         last_pdts_path = "{}_{}products_{}.json".format(output_file[:-4], nb_base_products,dataset)
         #output_file = '{}main_1blocks_10cells_{}products.pdt'.format(base_path,nb_base_products) 

   if os.path.isfile(output_file):
      print("Skipping initial PLEDGE run")
   else:
      print("Initial PLEDGE run")
      run_pledge(input_file,nb_base_products, output_file)
   
   product_set, last_population = extract_leaves(output_file)

   if os.path.isfile(last_pdts_path):
      print("Skipping initial training")
      f1 = open(last_pdts_path, 'r')
      vects = json.loads(f1.read())
      last_population["products"] = [KerasFeatureVector.from_vector(vect) for vect in vects]

   else:
      pop = train_products(product_set, output_file, dataset)
      last_population["products"] = last_population["products"]+ pop
      f1 = open(last_pdts_path, 'w')
      f1.write(json.dumps([i.to_vector() for i in last_population["products"]]))
      f1.close()   
   
   #list of (mutant_ratio,mutant_prds) sum of count_ratio should equal 1
   mutant_ratios = ((1, 0.2),  (0.5, 0.1), (0.5, 0.1), (0.5, 0.1), (0.5, 0.1), (0.5, 0.1), (0.25, 0.1), (0.25, 0.1), (0.25, 1))
   mutant_ratios = ((0.5, 1),)

   for evo in range(evolution_epochs):
      print("### evolution epoch {}".format(evo+last_evolution_epoch))
      new_pop, new_pop_labels = select(last_population, survival_count)
      last_population["products"] = new_pop
      print("### remaining top individuals {}".format(len(new_pop)))
      for i in range(survival_count):

         last_pdts_paths = []
         print("### batch of product parent {}".format(i))
         
         for ratio_id, (ratio, nb_prd) in enumerate(mutant_ratios):
            last_pdts_paths.append(mutate("{}_epoch{}_mutant{}_ratio{}".format(dataset, evo,i, ratio_id),new_pop_labels[i],input_file, nb_prd*nb_product_perparent,ratio))
         
         for mutants_pdts_path in last_pdts_paths:
            print("### batch of pledge sub-products {}".format(mutants_pdts_path))
            product_set.load_products_from_url(mutants_pdts_path)
            pop = train_products(product_set, mutants_pdts_path, dataset)
            pop = sorted(pop, key=lambda x: x.accuracy, reverse=True)
            f1 = open(mutants_pdts_path[:-4]+".json", 'w')
            
            f1.write(json.dumps([i.to_vector() for i in pop]))
            f1.close()   

            last_population["products"] = last_population["products"] + pop


      print("### remaining total individuals {}".format(len(last_population["products"])))

      end = last_pdts_path.find("products")
      if end > -1:
         pdt_path = "{}_{}.json".format(last_pdts_path[0:end],evo )
      else:
         pdt_path = "{}_{}products_{}.json".format(last_pdts_path[:-4], nb_base_products,evo)

      f1 = open(pdt_path, 'w')
      pop = sorted(last_population["products"], key=lambda x: x.accuracy, reverse=True)
      f1.write(json.dumps([i.to_vector() for i in pop]))
      f1.close()  




def main(argv):
   input_file = ''
   output_file = ''
   products_file = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:p:",["ifile=","ofile=","pfile="])
   except getopt.GetoptError:
      pass
   for opt, arg in opts:
      if opt == '-h':
         print('evolution.py -i <input_file> -o <output_file> -p <products_file>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         input_file = arg
      elif opt in ("-o", "--ofile"):
         output_file = arg
      elif opt in ("-p", "--pfile"):
         products_file = arg

   run(input_file, output_file, last_pdts_path=products_file, dataset="cifar")
   


if __name__ == "__main__":
   main(sys.argv[1:])