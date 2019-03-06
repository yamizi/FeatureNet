# -*- coding: utf-8 -*-
""""""
from __future__ import absolute_import, division, print_function, unicode_literals
from model.keras_model import KerasFeatureModel
from keras.datasets import mnist, cifar10
import keras
from keras import backend as K

import json
import time
from keras.callbacks import Callback

class TimedStopping(Callback):
    '''Stop training when enough time has passed.
    # Arguments
        seconds: maximum time before stopping.
        verbose: verbosity mode.
    '''
    def __init__(self, generator,epoch_seconds=None, total_seconds=None, verbose=0):
        super(TimedStopping, self).__init__()

        self.start_time = 0
        self.epoch_seconds = epoch_seconds
        self.total_seconds = total_seconds
        self.verbose = verbose
        self.generator = generator

    def on_train_begin(self, logs={}):
        self.start_time = time.time()

    def on_epoch_begin(self, epoch, logs={}):
        self.start_epoch = time.time()

    def on_epoch_end(self, epoch, logs={}):
        if self.total_seconds and time.time() - self.start_time > self.total_seconds:
            self.generator.stop_training=self.model.stop_training = True
            if self.verbose:
                print('Stopping after total time reached %s seconds.' % self.total_seconds)

        if self.epoch_seconds and time.time() - self.start_epoch > self.epoch_seconds:
            self.generator.stop_training=self.model.stop_training = True
            if self.verbose:
                print('Stopping after epoch time reached %s seconds.' % self.epoch_seconds)
        

class TensorflowGenerator(object):
    accuracy = 0
    stop_training = False
    def __init__(self, product, epochs=12, dataset="mnist"):
        
        if product:
            batch_size = 128
            num_classes = 10

            # the data, split between train and test sets
            if dataset=="mnist":
                (x_train, y_train), (x_test, y_test) = mnist.load_data()
            elif dataset=="cifar":
                (x_train, y_train), (x_test, y_test) = cifar10.load_data()

            # input image dimensions
            img_rows, img_cols, channels = x_train.shape[1], x_train.shape[2], x_train.shape[3] if len(x_train.shape) ==4 else 1

            # convert class vectors to binary class matrices
            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.to_categorical(y_test, num_classes)

            if K.image_data_format() == 'channels_first':
                x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
                x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
                input_shape = (channels, img_rows, img_cols)
            else:
                x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
                x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
                input_shape = (img_rows, img_cols, channels)


            model = None
            model =KerasFeatureModel.parse_feature_model(product)
            

            print("====> Loading new feature model with {0} blocks".format(len(model.blocks)))
            self.model = model.build(input_shape, num_classes)

            if not self.model:
                return 

            

            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            x_train /= 255
            x_test /= 255
            print('x_train shape:', x_train.shape)
            print(x_train.shape[0], 'train samples')
            print(x_test.shape[0], 'test samples')


            timed = TimedStopping(self,None, 600)
            begin_training = time.time()
            self.model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test), callbacks=[timed])
            
            self.training_time = time.time() - begin_training
            score = self.model.evaluate(x_test, y_test, verbose=0)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])
            print('model params:', self.model.count_params())
                

            self.accuracy = score[1]
            self.params = self.model.count_params()

    def load_products(self, product):
        def build_rec(node, level=0):
            print("-"*level + node.get("label"))
            for child in node.get("children"):
                build_rec(child, level+1)

        build_rec(product)
        

#f = open("./lenet5.json", "r")

#product = json.loads(f.read())

#tensorflow = TensorflowGenerator(product)

