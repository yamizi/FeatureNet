# -*- coding: utf-8 -*-
""""""
from __future__ import absolute_import, division, print_function, unicode_literals
from model.keras_model import KerasFeatureModel
import json


class TensorflowGenerator(object):
    def __init__(self, product):
        
        if product:
            KerasFeatureModel.parse_feature_model(product)

    def load_products(self, product):
        def build_rec(node, level=0):
            print("-"*level + node.get("label"))
            for child in node.get("children"):
                build_rec(child, level+1)

        build_rec(product)
        

def parse_products():

    from products_tree import ProductSet
    url = "C:/Users/salah.ghamizi/Downloads/pledge-master/tensorflow//NAS Products.pdt"
    productSet = ProductSet(url)
    print("number of features {0}, number of products: {1}".format(productSet.nbFeatures, productSet.nbProducts))    
    product = productSet.format_product(2)

    f = open("./block.json", "w")
    f.write(json.dumps([product[0]])) 

#parse_products()

f = open("./block.json", "r")

tensorflow = TensorflowGenerator(json.loads(f.read()))