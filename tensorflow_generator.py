# -*- coding: utf-8 -*-
""""""
from __future__ import absolute_import, division, print_function, unicode_literals
from model.keras_model import KerasFeatureModel
from keras.datasets import mnist, cifar10, cifar100
import keras
from keras import backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import json
import time
from keras.callbacks import Callback, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from keras.optimizers import Adam

#from keras.utils.training_utils import multi_gpu_model
#from tensorflow.python.client import device_lib


# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_flops():
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    #print('Learning rate: ', lr)
    return lr
    
def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()

    # use the same config as you used to create the session
    config = tf.ConfigProto() #allow_soft_placement=True, log_device_placement=True)
    set_session(tf.Session(config=config))

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
    model_graph = ""
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
    default_batchsize = 64
    num_classes = 10

    def __init__(self, product, epochs=12, dataset="mnist", data_augmentation = False, depth=1, product_features=None, features_label=None, no_train=False,clear_memory=True, batch_size=128):
        #product_features is a list of enabled and disabled features based on the original feature model
        
        if batch_size ==0:
            batch_size = TensorflowGenerator.default_batchsize

        if clear_memory:
            reset_keras()
            
        if product:
            TensorflowGenerator.init_dataset(dataset, data_augmentation)

            timed = TimedStopping(self,None, 6000)
            begin_training = time.time()    
            self.model =KerasFeatureModel.parse_feature_model(product, name="", depth=depth, product_features=product_features, features_label=features_label)

            print("====> Loading new feature model with {0} blocks".format(len(self.model.blocks)))
            model = self.model.build(TensorflowGenerator.input_shape, TensorflowGenerator.num_classes)
            
            if not model:
                print("#### model is not valid ####")
                return 

            optimizers = [  Adam(lr=lr_schedule(0)), "sgd"]
            losss = ['categorical_crossentropy']
            print("Compile Tensorflow model with loss:{}, optimizer {}".format(losss[0], optimizers[0]))
            model.compile(loss=losss[0], metrics=['accuracy'], optimizer=optimizers[0])

            self.model.nb_params = self.params = model.count_params()
            print('model blocks,layers,params,flops: {} '.format(self.model.to_kerasvector()))
            
            if no_train:
                self.print()
                return  
                
            early_stopping = EarlyStopping(monitor='val_acc', mode='max', min_delta=0.005, patience=100)
            lr_scheduler = LearningRateScheduler(lr_schedule)

            lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                cooldown=0,
                                patience=5,
                                min_lr=0.5e-6)
                                
            callbacks=[timed, early_stopping, lr_reducer, lr_scheduler]
            print("training with batch size {} epochs {} callbacks {} dataset {} data-augmentation {}".format(batch_size,epochs, callbacks,dataset , data_augmentation))
            
            history = model.fit(TensorflowGenerator.X_train, TensorflowGenerator.Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=0,
                    validation_data=(TensorflowGenerator.X_test, TensorflowGenerator.Y_test), 
                    callbacks=callbacks)
            
            self.training_time = time.time() - begin_training
            score = model.evaluate(TensorflowGenerator.X_test, TensorflowGenerator.Y_test, verbose=0)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])
            
                
            #self.model.nb_flops =  self.flops = get_flops()
            self.model.accuracy = self.accuracy = score[1]
            self.history = (history.history['acc'], history.history['val_acc'])

    @staticmethod
    def init_dataset(dataset, data_augmentation=False):
        TensorflowGenerator.num_classes = 10

        if TensorflowGenerator.dataset != dataset:

            # the data, split between train and test sets
            if dataset=="mnist":
                (x_train, y_train), (x_test, y_test) = mnist.load_data()
            elif dataset=="cifar":
                (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            elif dataset=="cifar100":
                (x_train, y_train), (x_test, y_test) = cifar100.load_data()
                TensorflowGenerator.num_classes = 100

            # input image dimensions
            img_rows, img_cols, channels = x_train.shape[1], x_train.shape[2], x_train.shape[3] if len(x_train.shape) ==4 else 1

            # convert class vectors to binary class matrices
            y_train = keras.utils.to_categorical(y_train, TensorflowGenerator.num_classes)
            y_test = keras.utils.to_categorical(y_test, TensorflowGenerator.num_classes)

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

    def print(self, include_summary=True, invalid_params=True, export_png=True):
        model = self.model.model
        if include_summary:
            model.summary()
            
        if invalid_params:
            missing_params = self.model.get_custom_parameters()
            for name,(node, params) in missing_params.items():
                print("{0}:{1}".format(name, params))
        
        if TensorflowGenerator.model_graph and export_png:
            from keras.utils import plot_model
            plot_model(model, to_file='{}.png'.format(TensorflowGenerator.model_graph))
            
    def load_products(self, product):
        def build_rec(node, level=0):
            #print("-"*level + node.get("label"))
            for child in node.get("children"):
                build_rec(child, level+1)

        build_rec(product)
        

#f = open("./lenet5.json", "r")

#product = json.loads(f.read())

#tensorflow = TensorflowGenerator(product)

