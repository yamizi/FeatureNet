# -*- coding: utf-8 -*-
""""""
from __future__ import absolute_import, division, print_function, unicode_literals

import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import numpy as np


class TensorflowGenerator(object):
    def __init__(self, product):
        
        if product:
            self.load_products(product)

    def load_products(self, product):
        model = Sequential()

        def build_rec(node, level=0):
            print("-"*level + node.get("label"))
            for child in node.get("children"):
                build_rec(child, level+1)

        build_rec(product)
        


from products_tree import ProductSet
url = "C:/Users/salah.ghamizi/Downloads/pledge-master/tensorflow//NAS Products.pdt"
productSet = ProductSet(url)
print("number of features {0}, number of products: {1}".format(productSet.nbFeatures, productSet.nbProducts))    
product = productSet.format_product(2)

tensorflow = TensorflowGenerator(product[0])