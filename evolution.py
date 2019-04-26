from tensorflow_generator import TensorflowGenerator
from products_tree import ProductSet
import multiprocessing
import json
import random
import sys, getopt
from model.keras_model import KerasFeatureVector

dataset = "cifar"
baseurl= "./pledge_product/" 
baseurl= "../products/"
base_products = "100Products" 
base_products = "100products_full_5x5"
nb_base_products = 4
training_epochs = 12
evolution_epochs = 20
survival_rate = 0.5

def sort_select(list_vectors, survival_rate=0.5):
    sorted_vectors = sorted(list_vectors, key=lambda x: x.fitness, reverse=True)
    return sorted_vectors[0:int(survival_rate*len(list_vectors))]


def evolve(initial_population, survival_rate=0.5):
    original_size = len(initial_population)
    top_population = sort_select(initial_population, survival_rate)
    top_size = len(top_population)
    current_population = [i for i in top_population]
    while len(current_population) < original_size:
        index1 = random.randint(0, top_size-1)
        index2 = (index1 + random.randint(1, top_size-2) ) %top_size if top_size>2 else index1
        
        new_vector = top_population[index1]
        new_vector = new_vector.cross_over(top_population[index2])
        new_vector.mutate()
        current_population.append(new_vector)

    return current_population


def train(prod,training_epochs, dataset, e, features):
        tensorflow = TensorflowGenerator(prod,training_epochs, dataset, features)
        e.accuracy = tensorflow.model.accuracy


def run(inputfile=None,outputfile=None):

      
   initial_product_set = ProductSet(baseurl+base_products+".pdt")
   initial_products_vectors = []
   last_evolution_epoch = 0

   if inputfile:
      f = open(inputfile+".json", 'r')
      line = f.readline()
      while line:
         line = f.readline()
      
      last_evol = line.split(" ")
      last_evolution_epoch = int(last_evol[0])
      initial_products_vectors = json.loads(last_evol[1])
      initial_products_vectors = [KerasFeatureVector.from_vector(i) for i in initial_products_vectors]
      f.close()

   else:
      inputfile = baseurl+base_products+"_initial"
      for index, (product, original_product) in enumerate(initial_product_set.format_products()):
         tensorflow = TensorflowGenerator(product,training_epochs, dataset, initial_product_set.features)
         initial_products_vectors.append(tensorflow.model.to_vector())

         if nb_base_products > 0 and index == nb_base_products:
            break


   last_population = initial_products_vectors[:nb_base_products] if nb_base_products else initial_products_vectors
   f1 = open(inputfile+".json", 'a')
   f1.write("\n{} {}".format(last_evolution_epoch+i,json.dumps([i.to_vector() for i in last_population])))
   f1.close()   
   
   for i in range(evolution_epochs):
      print("### evolution epoch {}".format(i+last_evolution_epoch))
      new_pop = evolve(last_population)
      print("evolved population {}, parent fitness {}".format(len(new_pop), [pop.fitness for pop in new_pop]))

      processes = []

      for e in new_pop:
         if not e.accuracy:
               prod, original_product  = initial_product_set.format_product(original_product=e.features)
               p = multiprocessing.Process(target=train, args=(prod,training_epochs, dataset, e, initial_product_set.features))
               p.start()
               processes.append(p)

      for p in processes:
         p.join()

      f1 = open(inputfile+".json", 'a')
      f1.write("\n{} {}".format(last_evolution_epoch+i,json.dumps([i.to_vector() for i in new_pop])))
      f1.close()   
      last_population = new_pop

   if not outputfile:
      outputfile = "{0}report_evol_{1}epochs_{2}evolution_{3}.txt".format(baseurl,dataset, training_epochs, evolution_epochs)

   f2 = open(outputfile,"a")
   f2.write("\r\n".join(str(x) for x in last_population))



def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      pass
   for opt, arg in opts:
      if opt == '-h':
         print('evolution.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg

   run(inputfile, inputfile)
   


if __name__ == "__main__":
   main(sys.argv[1:])