# -*- coding: utf-8 -*-
""""""
from __future__ import absolute_import, division, print_function, unicode_literals
from model.keras_model import KerasFeatureModel
from keras.datasets import mnist, cifar10
import keras
from keras import backend as K
import tensorflow as tf
import json
import time
from keras.callbacks import Callback, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


def get_flops():
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops

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
    training_time = 0
    params = 0
    flops = 0
    stop_training = False
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    history = ([],[])
    dataset = None
    input_shape = (0,0,0)

    def __init__(self, product, epochs=12, dataset="mnist", data_augmentation = False, depth=1, features=None):
        #features is a list of enabled and siabled features based on the original feature model
        if product:
            batch_size = 128 #64
            num_classes = 10

            if TensorflowGenerator.dataset != dataset:

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
                    TensorflowGenerator.input_shape = (channels, img_rows, img_cols)
                else:
                    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
                    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
                    TensorflowGenerator.input_shape = (img_rows, img_cols, channels)

                x_train = x_train.astype('float32')
                x_test = x_test.astype('float32')
                x_train /= 255
                x_test /= 255
                #print('x_train shape:', x_train.shape)
                #print(x_train.shape[0], 'train samples')
                #print(x_test.shape[0], 'test samples')

                if data_augmentation:

                    augment_size=5000
                    train_size = x_train.shape[0]

                    datagen = ImageDataGenerator(
                    rotation_range=10,
                    zoom_range = 0.05, 
                    width_shift_range=0.07,
                    height_shift_range=0.07,
                    horizontal_flip=False,
                    vertical_flip=False, 
                    data_format="channels_last",
                    zca_whitening=True)

                    # compute quantities required for featurewise normalization
                    # (std, mean, and principal components if ZCA whitening is applied)
                    datagen.fit(x_train, augment=True)

                    randidx = np.random.randint(train_size, size=augment_size)
                    x_augmented = x_train[randidx].copy()
                    y_augmented = y_train[randidx].copy()
                    
                    x_augmented = datagen.flow(x_augmented, np.zeros(augment_size), batch_size=augment_size, shuffle=False).next()[0]
                    x_train = np.concatenate((x_train, x_augmented))
                    y_train = np.concatenate((y_train, y_augmented))

                TensorflowGenerator.X_train = x_train
                TensorflowGenerator.X_test = x_test
                TensorflowGenerator.Y_train = y_train
                TensorflowGenerator.Y_test = y_test
                TensorflowGenerator.dataset = dataset

    
            timed = TimedStopping(self,None, 6000)
            begin_training = time.time()    
            model =KerasFeatureModel.parse_feature_model(product, name="", depth=depth, features=features)

            print("====> Loading new feature model with {0} blocks".format(len(model.blocks)))
            self.model = model.build(TensorflowGenerator.input_shape, num_classes)

            if not self.model:
                print("#### model is not valid ####")
                return 
                
            
            early_stopping = EarlyStopping(monitor='val_acc', mode='max', min_delta=0.01, patience=25)

            history = self.model.fit(TensorflowGenerator.X_train, TensorflowGenerator.Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(TensorflowGenerator.X_test, TensorflowGenerator.Y_test), 
                    callbacks=[timed, early_stopping])
            
            self.training_time = time.time() - begin_training
            score = self.model.evaluate(TensorflowGenerator.X_test, TensorflowGenerator.Y_test, verbose=0)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])
            print('model params:', self.model.count_params())
            
                
            model.nb_flops =  self.flops = get_flops()
            model.nb_params = self.params = self.model.count_params()
            model.score = self.accuracy = score[1]
            self.history = (history.history['acc'], history.history['val_acc'])

    def load_products(self, product):
        def build_rec(node, level=0):
            #print("-"*level + node.get("label"))
            for child in node.get("children"):
                build_rec(child, level+1)

        build_rec(product)
        

#f = open("./lenet5.json", "r")

#product = json.loads(f.read())

#tensorflow = TensorflowGenerator(product)

