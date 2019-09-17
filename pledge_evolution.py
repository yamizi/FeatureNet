import subprocess
import sys
import getopt
import json
import os
from tensorflow_generator import TensorflowGenerator
from model.keras_model import KerasFeatureVector
from products_tree import ProductSet, ProductSetError
import random, math

import tensorflow
import gc
import datetime
import time
from math import ceil
import re


pledge_path = '../products/PLEDGE.jar'
base_path = '../products'
base_training_epochs = 25



def reset_keras(classifier=None):
    
    if classifier:
        try:
            del classifier
        except:
            pass

    # if it's done something you should see a number being outputted
    print("cleaning memory {}".format(gc.collect()))

def run_pledge(input_file, nb_base_products, output_file, duration=60):
    params = ['java', '-jar', pledge_path, 'generate_products']
    params = params+['-fm', input_file, '-nbProds',
                    str(nb_base_products), '-o', output_file, '-timeAllowedMS',str(duration*1000)]
    start = time.time()
    print("running pledge on {}: {}".format(
        params, datetime.datetime.now()))
    pledge_result = subprocess.check_call(params)

    end = time.time()
    print("pledge result {} in {}s".format(pledge_result, str(end-start)))
    return pledge_result

def default_pledge_output(base_path, nb_base_products):
    output_file = '{}/{}products.pdt'.format(base_path, nb_base_products)
    return output_file

class PledgeEvolution(object):

    attacks = ["cw"]

    @staticmethod
    def select(last_population, survival_count):
        labels = last_population.get("filtered_features_label")
        products = last_population.get("products")
        products = sorted(products, key=lambda x: x.accuracy,
                        reverse=True)[:survival_count]

        products_labels = [[labels[k] for k in labels.keys(
        ) if prd.features[int(k)-1] == 1] for prd in products]
        return products, products_labels

    @staticmethod
    def generate_children(mutant_path, mutant_labels, initial_feature_model, nb_products, constraints_ratio=0.5):

        mutant_labels = [
            mutant for mutant in mutant_labels if random.random() < constraints_ratio]

        print("### mutate product from {} to {}".format(initial_feature_model, mutant_path))
        import xml.etree.ElementTree as ET
        tree = ET.parse(initial_feature_model)
        root = tree.getroot()
        constraints = root.getchildren()[1]
        constraints_raw = constraints.text.split("\n")
        nb_constraints = len(constraints_raw)
        constraints_raw = constraints_raw[:-1]
        for i, lbl in enumerate(mutant_labels):
            constraints_raw.append(
                "C{}:~Architecture  or  ~{}".format(nb_constraints+i-1, lbl))

        constraints.text = "\n".join(constraints_raw)
        dst = "{}.xml".format(mutant_path)
        tree.write(dst, encoding="UTF-8", xml_declaration=True)

        if not os.path.isfile(dst):
            print("file {} not saved yet".format(dst))
            raise Exception()

        output_file = "{}.{}".format(dst[:-4], "pdt")
        run_pledge(dst, int(nb_products), output_file)

        # possibility to generate multiple products each containing a different set of constraints
        return output_file

    @staticmethod
    def extract_leaves(output_file):
        initial_product_set = ProductSet(output_file,binary_products=True)
        filtered_features_label = ProductSet.filter_leaves(
            initial_product_set.features)
        last_population = {"filtered_features_label": [], "products": []}
        last_population["filtered_features_label"] = filtered_features_label

        return initial_product_set, last_population

    @staticmethod
    def train_products(initial_product_set, dataset,training_epochs, max_products=0, export_path=""):
        start = time.time()
        print("### training products for dataset {}: {}".format(
            dataset, datetime.datetime.now()))
        last_population = []
        for index, (product, original_product) in enumerate(initial_product_set.format_products()):
            print("### training product {}".format(index))
            tensorflow = TensorflowGenerator(product, training_epochs, dataset, product_features=original_product, depth=1,
                                            features_label=initial_product_set.features, no_train=False, data_augmentation=False,eval_robustness=PledgeEvolution.attacks, save_path=export_path)

            if tensorflow and hasattr(tensorflow,"model") and tensorflow.model:
                last_population.append(tensorflow.model.to_kerasvector())
            reset_keras(tensorflow)

            if max_products>0 and index==max_products:
                break

        end = time.time()
        print("### training products over, took {}s".format(str(end-start)))
        return last_population

    @staticmethod
    def evolve(survival_count,mutant_ratios, evo, session_path, new_pop_labels,input_file, nb_product_perparent, product_set, dataset, last_population, training_epochs  ):
        for i in range(survival_count):

            print("### batch of product parent {}".format(i))

            for batch_id, (ratio, nb_prd) in enumerate(mutant_ratios):
                mutant_id = "e{}_m{}_b{}".format(evo, i, batch_id)
                mutant_path = "{}/{}".format(session_path, mutant_id)
                mutants_pdts_path = PledgeEvolution.generate_children(mutant_path, new_pop_labels[i], input_file, max(3,ceil(nb_prd*nb_product_perparent)), ratio)
                print("### batch of pledge sub-products {}".format(mutants_pdts_path))
                try:
                    product_set.load_products_from_url(mutants_pdts_path)
                except ProductSetError as e:
                    print(e)
                    continue

                pop = PledgeEvolution.train_products(product_set, dataset,training_epochs, export_path=mutant_path)
                pop = sorted(pop, key=lambda x: x.accuracy, reverse=True)
                f1 = open("{}.json".format(mutant_path), 'w')

                f1.write(json.dumps([i.to_vector() for i in pop]))
                f1.close()

                last_population["products"] = last_population["products"] + pop

        return last_population

    @staticmethod
    def run(base_path, input_file="", output_file="", last_pdts_path="", nb_base_products=100, dataset="cifar", training_epochs=25, evolution_epochs = 50):

        if not os.path.isdir(base_path):
            os.mkdir(base_path)

        session_path = "{}/{}".format(base_path, dataset)

        if not os.path.isdir(session_path):
            os.mkdir(session_path)

        survival_rate = 0.05
        survival_count = math.ceil(survival_rate*nb_base_products)
        nb_product_perparent =  int((nb_base_products-survival_count) / survival_count)
        last_evolution_epoch = 0
        reset_keras()

        if not input_file:
            input_file = '{}/nas_2_5.xml'.format(base_path)
            #input_file = '{}main_1block_10_cells.xml'.format(base_path)

        if not output_file:
            output_file = default_pledge_output(base_path, nb_base_products)
            #output_file = '{}main_1blocks_10cells_{}products.pdt'.format(base_path,nb_base_products)

        if not last_pdts_path:
            p = output_file.find("products")
            last_pdts_path = "{}/{}products.json".format(
                session_path, nb_base_products)
            #output_file = '{}main_1blocks_10cells_{}products.pdt'.format(base_path,nb_base_products)

        if os.path.isfile(output_file):
            print("Skipping initial PLEDGE run, file found in {}".format(output_file))
        else:
            print("Initial PLEDGE run")
            run_pledge(input_file, nb_base_products, output_file)

        product_set, last_population = PledgeEvolution.extract_leaves(output_file)

        ## last_population["products"] is a list of KerasFeatureVector objects
        if os.path.isfile(last_pdts_path):
            print("Skipping initial training")
            f1 = open(last_pdts_path, 'r')
            vects = json.loads(f1.read())
            last_population["products"] = [
                KerasFeatureVector.from_vector(vect) for vect in vects]

            pattern = 'products_e(\d+).json'
            result = re.findall(pattern, last_pdts_path) 
            if len(result):
                last_evolution_epoch = int(result[0])+1
                
        else:
            export_path = "{}/initial".format(session_path)
            pop = PledgeEvolution.train_products(product_set, dataset, training_epochs, nb_base_products, export_path=export_path)
            last_population["products"] = last_population["products"] + pop
            f1 = open(last_pdts_path, 'w')
            f1.write(json.dumps([i.to_vector()
                                for i in last_population["products"]]))
            f1.close()

        # list of (mutant_ratio,mutant_prds) sum of count_ratio should equal 1
        #mutant_ratios = [(1, 0.2),  (0.5, 0.2), (0.5, 0.2), (0.5, 0.2), (0.25, 0.2)]
        #mutant_ratios = [(1, 0.2),  (0.5, 0.3),(0.5, 0.3), (0.25, 0.2)]
        
        mutant_ratios = [(0.2, 0.5),(0.2, 0.5)]

        for evo in range(evolution_epochs):
            # The more we increase the epochs, the more we keep all the leaves
            mutant_ratios[1] = mutant_ratios[0] = (0.2+ 0.75*evo/evolution_epochs,0.5) 

            print("### evolution epoch {}".format(evo+last_evolution_epoch))
            new_pop, new_pop_labels = PledgeEvolution.select(last_population, survival_count)
            last_population["products"] = new_pop
            print("### remaining top individuals {}".format(len(new_pop)))
            
            last_population = PledgeEvolution.evolve(survival_count,mutant_ratios, evo, session_path, new_pop_labels,input_file, nb_product_perparent, product_set, dataset, last_population , training_epochs )
            pop = sorted(last_population["products"],
                        key=lambda x: x.accuracy, reverse=True)
        
            pdt_path = "{}/{}products_e{}.json".format(
                session_path, nb_base_products, evo)
            print("### remaining total individuals {} saved to {}. top accuracy: {}".format(
                len(pop),pdt_path, pop[0].accuracy))
            f1 = open(pdt_path, 'w')
            f1.write(json.dumps([i.to_vector() for i in pop]))
            f1.close()

        return session_path

    @staticmethod
    def end2end(base_path, nb_base_products, input_file="", output_file="", last_pdts_path="", dataset="cifar", training_epochs=25):
        from extender import generate_featuretree

        if not os.path.isdir(base_path):
            os.mkdir(base_path)

        _input_file = "main_1block_nas.xml"
        print("End to end NAS Search from {} to {} products".format(_input_file, nb_base_products))

        _nb_blocks,_nb_cells, _nb_products = nb_base_products
        
        full_fm_file = input_file if input_file else "{}/nas_{}.xml".format(base_path,"_".join([str(e) for e in nb_base_products]))

        if os.path.isfile(output_file):
            print("Skipping full FM generation, file found in {}".format(full_fm_file))
        else:
            generate_featuretree(_input_file,full_fm_file,int(_nb_cells),int(_nb_blocks))

        return full_fm_file



def main(argv):
    input_file = ''
    output_file = ''
    products_file = ''
    base = base_path
    nb_base_products=[100]
    dataset = "cifar"
    training_epochs = base_training_epochs
    
    try:
        opts, args = getopt.getopt(argv, "hn:d:b:i:o:p:t:", [
                                   "nb=","dataset=", "bpath=", "ifile=", "ofile=", "pfile=", "training_epoch="])
    except getopt.GetoptError:
        pass
    print("arguments {}".format(opts))
    for opt, arg in opts:
        
        if opt == '-h':
            print(
                'pledge_evolution.py -n <nb_architectures> -d <dataset> -b <base_path> -i <input_file> -o <output_file> -p <products_file> -t <training_epoch>')
            sys.exit()
        elif opt in ("-n", "--nb"):
            nb_base_products = arg.split("x")
        elif opt in ("-d", "--dataset"):
            dataset = arg
        elif opt in ("-b", "--bpath"):
            base = arg
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-o", "--ofile"):
            output_file = arg
        elif opt in ("-p", "--pfile"):
            products_file = arg
        elif opt in ("-t", "--training_epoch"):
            training_epochs = int(arg)
        
    if len(nb_base_products) ==1:
        PledgeEvolution.run(base, input_file, output_file,
        last_pdts_path=products_file, dataset=dataset, nb_base_products=int(nb_base_products[0]), training_epochs=training_epochs)
    else:
        _nb_blocks,_nb_cells, _nb_products = nb_base_products
        
        full_fm_file = PledgeEvolution.end2end(base, nb_base_products, input_file, output_file,
        last_pdts_path=products_file, dataset=dataset, training_epochs=training_epochs )
        PledgeEvolution.run(base_path, full_fm_file, output_file,last_pdts_path=products_file, dataset=dataset, nb_base_products=int(_nb_products), training_epochs=training_epochs)



if __name__ == "__main__":
    main(sys.argv[1:])
