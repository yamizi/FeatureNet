# -*- coding: utf-8 -*-
""""""
from __future__ import absolute_import, division, print_function, unicode_literals
from model.keras_model import KerasFeatureModel
from keras.datasets import mnist
import keras
from keras import backend as K

import json


class TensorflowGenerator(object):
    def __init__(self, product):
        
        if product:
            model =KerasFeatureModel.parse_feature_model(product)
            self.model = model.build((28,28,1))

            
            batch_size = 128
            num_classes = 10
            epochs = 12

            # input image dimensions
            img_rows, img_cols = 28, 28

            # the data, split between train and test sets
            (x_train, y_train), (x_test, y_test) = mnist.load_data()

            if K.image_data_format() == 'channels_first':
                x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
                x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
                input_shape = (1, img_rows, img_cols)
            else:
                x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
                x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
                input_shape = (img_rows, img_cols, 1)

            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            x_train /= 255
            x_test /= 255
            print('x_train shape:', x_train.shape)
            print(x_train.shape[0], 'train samples')
            print(x_test.shape[0], 'test samples')

            # convert class vectors to binary class matrices
            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.to_categorical(y_test, num_classes)

            tensorflow.model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
            score = tensorflow.model.evaluate(x_test, y_test, verbose=0)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])


    def load_products(self, product):
        def build_rec(node, level=0):
            print("-"*level + node.get("label"))
            for child in node.get("children"):
                build_rec(child, level+1)

        build_rec(product)
        

f = open("./lenet5.json", "r")

product = json.loads(f.read())

tensorflow = TensorflowGenerator(product)

