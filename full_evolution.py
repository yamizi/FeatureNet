import json, pickle
import os
from tensorflow_generator import TensorflowGenerator
from model.keras_model import KerasFeatureVector, KerasFeatureModel
from model.mutation.mutable_base import MutableBase, MutationStrategies
from products_tree import ProductSet, ProductSetError
import random, math
from numpy.random import choice
import numpy as np
import tensorflow
import gc
import datetime
import time
from math import ceil
import re
import copy 



def reset_keras(classifier=None):
    
    if classifier:
        try:
            del classifier
        except:
            pass

    # if it's done something you should see a number being outputted
    print("cleaning memory {}".format(gc.collect()))

class FullEvolution(object):

    
    @staticmethod
    def select(last_population, survival_count):
        fittest = []
        x =  [e.accuracy for e in last_population]
        e_x = np.exp(x - np.max(x))
        last_population_probability =  e_x / e_x.sum()

        last_population_size = len(last_population)
        
        # We keep the top individuals + randomly picked with probability distribution
        elitist_count = math.ceil(survival_count/4)
        for i in range(min(last_population_size,elitist_count)):
            individual= last_population[i]
            fittest.append(individual)
        
        for i in range(min(last_population_size,survival_count-elitist_count)):
            individual= choice(last_population, None, last_population_probability.tolist())
            fittest.append(individual)
        
        return fittest

    @staticmethod
    def generate_mutant(parent, mutation_ratio):
        
        nb_max_mutations = 100
        if MutableBase.mutation_stategy==MutationStrategies.CHOICE :
            mutations = np.random.uniform(size=nb_max_mutations)
            nb_mutations = len([e for e in mutations if e<mutation_ratio])
        else:
            nb_mutations = 1
        blocks = parent.dump_blocks()
        blocks = copy.deepcopy(blocks)
        mutant = KerasFeatureModel.parse_blocks({"blocks":blocks["blocks"]})

        for j in range(nb_mutations):
            mutant.mutate(mutation_ratio)

        return mutant

        
    @staticmethod
    def train_products(initial_product_set, dataset,training_epochs, max_products=0):
        start = time.time()
        print("### training products for dataset {}: {}".format(
            dataset, datetime.datetime.now()))
        last_population = []
        for index, (product, original_product) in enumerate(initial_product_set.format_products()):
            print("### training product {}".format(index))
            tensorflow = TensorflowGenerator(product, training_epochs, dataset, product_features=original_product, depth=1,
                                            features_label=initial_product_set.features, no_train=False, data_augmentation=False)

            if tensorflow and hasattr(tensorflow,"model") and tensorflow.model:
                last_population.append(tensorflow.model.to_kerasvector())
            reset_keras(tensorflow)

            if max_products>0 and index==max_products:
                break

        end = time.time()
        print("### training products over, took {}s".format(str(end-start)))
        return last_population

    @staticmethod
    def evolve(evo, session_path, nb_product_perparent, dataset, new_pop, training_epochs, mutation_ratio=0.1, breed=True  ):
        len_pop = len(new_pop)
        mutants = []
        for i in range(len_pop):
            individual1 = new_pop[i]
            print("### generating children of product {}".format(individual1._name))
            for i in range(nb_product_perparent):
                if breed: 
                    individual2 = choice(new_pop)
                    individual = individual1.breed(individual2)
                else:
                    individual = individual1
                mutant = FullEvolution.generate_mutant(individual,mutation_ratio)
                mutants.append(mutant)    

        return new_pop + mutants

    @staticmethod
    def run(base_path, last_pdts_path="",nb_base_products=100, dataset="cifar", training_epochs=25,mutation_rate = 0.1,survival_rate = 0.1, breed=True, evolution_epochs=50):

        if not os.path.isdir(base_path):
            os.mkdir(base_path)

        session_path = "{}/{}".format(base_path, dataset)

        if not os.path.isdir(session_path):
            os.mkdir(session_path)

        session_path = "{}/ee{}_te{}_mr{}_sr{}".format(session_path,evolution_epochs,training_epochs,mutation_rate,survival_rate)

        if os.path.isdir(session_path) and not os.path.isfile(last_pdts_path):
            session_path = "{}_{}".format(session_path, int(time.time()))
        
        os.mkdir(session_path)
            
            
        survival_count = max(3,math.ceil(survival_rate*nb_base_products))
        nb_product_perparent =  ceil((nb_base_products-survival_count) / survival_count)
        last_evolution_epoch = 0
        reset_keras()

        
        if os.path.isfile(last_pdts_path):
            print("Resuming training")
            f1 = open(last_pdts_path, 'r')
            last_population= pickle.load(f1)
            last_population = [KerasFeatureModel.parse_blocks(e) for e in last_population]

            pattern = 'products_e(\d+).pickled'
            result = re.findall(pattern, last_pdts_path) 
            if len(result):
                last_evolution_epoch = int(result[0])+1
                
        else:
            tensorflow_gen = TensorflowGenerator("lenet5",training_epochs, dataset)
            last_population = [tensorflow_gen.model]

        for evo in range(evolution_epochs):
            print("### evolution epoch {}".format(evo+last_evolution_epoch))

            new_pop = FullEvolution.select(last_population, survival_count)
            
            mutant_population = FullEvolution.evolve(evo, session_path, nb_product_perparent, dataset, new_pop , training_epochs, mutation_ratio=mutation_rate, breed=breed )
            
            for index,model in enumerate(mutant_population):
                if model.accuracy==0:
                    #we do not train individuals preserved from previous generation
                    keras_model = TensorflowGenerator.build(model,dataset)
                    if not keras_model:
                        print("#### model is not valid ####")
                    else: 
                        TensorflowGenerator.train(model, training_epochs, TensorflowGenerator.default_batchsize, False,dataset)
                        TensorflowGenerator.eval_robustness(model)

                        path = "{}/e{}_{}".format(session_path, evo,model._name)

                        TensorflowGenerator.export_png(keras_model, path)

                pdt_path = "{}/e{}.json".format(
                    session_path, evo)
                
                f1 = open(pdt_path, 'a')
                vect = model.to_kerasvector().to_vector()
                f1.write("\r\n{} {}:{}".format(index,int(time.time()), json.dumps(vect)))
                f1.close()

            last_population = [x for x in mutant_population if x.accuracy>0.1]
            last_population = sorted(last_population,
                        key=lambda x: x.accuracy, reverse=True)
        
            pdt_path = "{}/{}products_e{}.pickled".format(
                session_path, nb_base_products, evo)
            print("### remaining total individuals {} saved to {}. top accuracy: {}".format(
                len(last_population),pdt_path, last_population[0].accuracy))
            f1 = open(pdt_path, 'w')
            #pickle.dump( [e.dump_blocks() for e in pop], f1)
            f1.close()


if __name__ == "__main__":
    input_file = ''
    output_file = ''
    products_file = ''
    base = '../products/local'
    nb_base_products=10
    dataset = "cifar"
    training_epochs = 1
    mutation_rate = 0.1
    survival_rate = 0.1
    breed = True
    evolution_epochs = 70

    MutableBase.MAX_NB_CELLS = 5
    MutableBase.MAX_NB_BLOCKS = 10
    FullEvolution.run(base, last_pdts_path=products_file, dataset=dataset, nb_base_products=nb_base_products, training_epochs=training_epochs, mutation_rate=mutation_rate,survival_rate=survival_rate, breed=breed, evolution_epochs=evolution_epochs)
    


