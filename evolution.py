from tensorflow_generator import TensorflowGenerator
from products_tree import ProductSet
import multiprocessing
import json
import random

dataset = "cifar"
baseurl= "./pledge_product/" 
baseurl= "../products/"
base_products = "100Products" 
base_products = "100products_full_5x5"
nb_base_products = 100
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


initial_product_set = ProductSet(baseurl+base_products+".pdt")
initial_products_vectors = []
for index, (product, features) in enumerate(initial_product_set.format_products()):
    tensorflow = TensorflowGenerator(product,training_epochs, dataset)
    initial_products_vectors.append(tensorflow.model.to_vector())

    if nb_base_products > 0 and index == nb_base_products:
        break


last_population = initial_products_vectors[:nb_base_products] if nb_base_products else initial_products_vectors
for i in range(evolution_epochs):
    print("### evolution epoch {}".format(i))
    new_pop = evolve(last_population)
    print("evolved population {}, parent fitness {}".format(len(new_pop), [pop.fitness for pop in new_pop]))

    for e in new_pop:
        if not e.accuracy:
            prod, original_product  = initial_product_set.format_product(original_product=e.features)
            tensorflow = TensorflowGenerator(prod,training_epochs, dataset)
            e.accuracy = tensorflow.model.accuracy

    last_population = new_pop


logpath = "{0}report_evol_{1}epochs_{2}evolution_{3}.txt".format(baseurl,dataset, training_epochs, evolution_epochs)
f2 = open(logpath,"a")
f2.write("\r\n".join(str(x) for x in last_population))