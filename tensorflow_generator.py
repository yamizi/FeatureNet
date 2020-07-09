# -*- coding: utf-8 -*-
""""""
from __future__ import absolute_import, division, print_function, unicode_literals
from model.keras_model import KerasFeatureModel
from keras.datasets import mnist, cifar10, cifar100, imdb
import keras
from keras import backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import json
import time
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from art.classifiers import KerasClassifier
from model import metrics
from keras.optimizers import Adam

from helpers import train_model

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
    num_words = 20000
    # cut texts after this number of words (among top max_features most common words)
    default_words_set_size = 80
    default_robustness_set_size = 500
    default_augmentation = False

    model_graph_export = True
    eval_metrics={}

    dataset_robustness = False
    datasets_classes = {"mnist":10,"cifar":10,"cifar10":10,"cifar100":100,"imdb_sentiment":1}
    datasets_output = {"mnist": "softmax", "cifar": "softmax", "cifar10": "softmax", "cifar100": "softmax", "imdb_sentiment": "sigmoid"}
    training_metrics = ['accuracy']
    training_loss = "categorical_crossentropy"


    def __init__(self, product, epochs=12, dataset="mnist", data_augmentation = False, depth=1, product_features=None, features_label=None, no_train=False,clear_memory=True, batch_size=128, eval_robustness=False, save_path=None, robustness_set_size=0, name="", optimizer=None):

        KerasFeatureModel.dataset_robustness = eval_robustness
        self.valid = True

        #product_features is a list of enabled and disabled features based on the original feature model
        if batch_size ==0:
            batch_size = TensorflowGenerator.default_batchsize
            
        if product:
            if isinstance(product, KerasFeatureModel):
                self.model = product
            else:
                self.model =KerasFeatureModel.parse_feature_model(product, name=name, depth=depth, product_features=product_features, features_label=features_label)

            print("====> Loading new feature model with {0} blocks".format(len(self.model.blocks)))
            model = TensorflowGenerator.build(self.model, dataset, clear_memory=clear_memory, optimizer=optimizer)

            if not model:
                print("#### model is not valid ####")
                self.valid = False
                return
            
            if no_train:
                return

            if save_path:
                save_path = "{}{}".format(save_path,self.model._name)
                
            history, training_time, score, keras_model = TensorflowGenerator.train(self.model, epochs, batch_size, dataset, data_augmentation,save_path=save_path)
            
            if not keras_model:
                print("#### model is not valid ####")
                self.valid = False
                return 
                
            if eval_robustness:
                TensorflowGenerator.eval_robustness(self.model, eval_robustness, robustness_set_size)

            if TensorflowGenerator.eval_metrics:
                for (m,f) in TensorflowGenerator.eval_metrics:
                    self.model.metrics[m] = f(keras_model)

            self.params = self.model.nb_params
            self.training_time = training_time
            self.accuracy = self.model.accuracy
            self.history = (history.history['acc'], history.history['val_acc'])

    @property
    def keras_model(self):
        return self.model.model

    @property
    def metrics(self):
        return self.model.metrics

    @staticmethod
    def add_metric(metric, func):
        TensorflowGenerator.eval_metrics[metric] = func
        
    @staticmethod
    def eval_attack_robustness(keras_model, attack_name, norm, robustness_set_size=0):
        
        attack_params = {"norm":norm}

        if attack_name=="cw":
            attack_params["targeted"] = False;
        elif attack_name=="pgd":
             attack_params["eps_step"] = 0.1
             attack_params["eps"]= 1.
        
        if robustness_set_size==0:
            adv_set = TensorflowGenerator.X_robustness
            y_set = TensorflowGenerator.Y_robustness
        else:
            adv_set = TensorflowGenerator.X_test[0:min(len(TensorflowGenerator.X_test), robustness_set_size)]
            y_set = TensorflowGenerator.Y_test[0:min(len(TensorflowGenerator.Y_test), robustness_set_size)]
            
        attack_robustness, adv_x = metrics.empirical_robustness(KerasClassifier(model=keras_model, clip_values=(0, 255)),adv_set,attack_name, attack_params)

        score_real = keras_model.evaluate(adv_set, y_set, verbose=0)
        score_adv = keras_model.evaluate(adv_x, y_set, verbose=0)

        return float(attack_robustness), score_real[1], score_adv[1]

    @staticmethod
    def eval_robustness(model, scores=[], robustness_set_size=0):
        keras_model = model.model
        if not keras_model or model.accuracy < 0.5:
            return 
        begin_robustness = time.time() 
        try:
            norm = 2
            r_l1 = 40
            r_l2 = 2
            r_li = 0.1
            nb_batches = 10
            batch_size = 5
            radius = r_l1 if norm==1 else (r_l2 if norm==2 else r_li)
            
            
            score_metrics = model.robustness_scores if not scores else scores
           
            if "clever" in score_metrics:
                if robustness_set_size==0:
                    x_set = TensorflowGenerator.X_robustness
                else:
                    x_set = TensorflowGenerator.X_test[0:robustness_set_size]
                scores = []
                art_model = KerasClassifier(model=keras_model, clip_values=(0, 255))
                for element in x_set:
                    score = metrics.clever_u(art_model, element, nb_batches, batch_size, radius, norm=norm, pool_factor=3)
                    scores.append(score)
                model.clever_score = np.average(scores)
            if "pgd" in score_metrics:
                model.pgd_score = TensorflowGenerator.eval_attack_robustness(keras_model, "pgd", norm,robustness_set_size)
            if "cw" in score_metrics:
                model.cw_score = TensorflowGenerator.eval_attack_robustness(keras_model, "cw", norm,robustness_set_size)
            if "fgsm" in score_metrics:
                model.fgsm_score = TensorflowGenerator.eval_attack_robustness(keras_model, "fgsm", norm,robustness_set_size)
            
        except Exception as e:
            import traceback
            print("error",e)
            print (traceback.format_exc())
        
        robustness_time = time.time() - begin_robustness
        model.robustness_score = getattr(model,"{}_score".format(scores[0]),[0]) if len(scores) else model.clever_score
        print('model robustness (clever, pgd, cw, fgsm): {} time:{}'.format((model.clever_score,model.pgd_score, model.cw_score, model.fgsm_score),robustness_time))

    @staticmethod
    def build(model, dataset, clear_memory=True, optimizer=None):

        if clear_memory:
            reset_keras()

        optimizer = optimizer if optimizer is not None else Adam(lr=lr_schedule(0))

        if dataset is not None:
            TensorflowGenerator.init_dataset(dataset)

        keras_model =  model.build(TensorflowGenerator.input_shape, TensorflowGenerator.datasets_classes.get(dataset),output_activation=TensorflowGenerator.datasets_output.get(dataset))

        if not keras_model:
            return keras_model

        keras_model.compile(loss=TensorflowGenerator.training_loss, metrics=TensorflowGenerator.training_metrics, optimizer=optimizer)

        model.nb_params =  keras_model.count_params()
        print('model blocks,layers,params,flops: {} '.format(model.to_kerasvector()))

        return keras_model


    @staticmethod
    def train(model, epochs, batch_size, dataset, data_augmentation=False, save_path=None):

        score = []

        if hasattr(model,"model"):
            keras_model = model.model
        else:
            keras_model = model
            
        begin_training = time.time()    
            
        model_path = "{}.h5".format(save_path) if save_path else None
            
        #print("training with batch size {} epochs {} callbacks {} dataset {} data-augmentation {}".format(batch_size,epochs, callbacks,dataset , data_augmentation))

        keras_model, history  = train_model(keras_model,TensorflowGenerator.X_train, TensorflowGenerator.Y_train,TensorflowGenerator.X_test, TensorflowGenerator.Y_test, epochs, batch_size, True, data_augmentation, model_path)

        training_time = time.time() - begin_training

        if keras_model:

            if model_path:
                #saving best model
                keras_model.save(model_path)
                TensorflowGenerator.export_png(keras_model, save_path)

            score = keras_model.evaluate(TensorflowGenerator.X_test, TensorflowGenerator.Y_test, verbose=0)
            
            #model.nb_flops = get_flops()
            model.accuracy =score[1]
            
            print('Test loss: {} Test accuracy: {} training_time {}'.format(score[0],  score[1], training_time))
        
        return history, training_time, score, keras_model

    @staticmethod
    def init_dataset(dataset, data_augmentation=False):
        TensorflowGenerator.num_classes = 10

        if TensorflowGenerator.dataset != dataset:

            if "imdb" in dataset:
                from keras.preprocessing import sequence
                (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=TensorflowGenerator.num_words)
                x_train = sequence.pad_sequences(x_train, maxlen=TensorflowGenerator.default_words_set_size)
                x_test = sequence.pad_sequences(x_test, maxlen=TensorflowGenerator.default_words_set_size)
                TensorflowGenerator.input_shape = (x_train.shape[1],)
                print('x_train shape:', TensorflowGenerator.input_shape)

            else:
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

            TensorflowGenerator.X_train = x_train
            TensorflowGenerator.X_test = x_test
            TensorflowGenerator.Y_train = y_train
            TensorflowGenerator.Y_test = y_test

            if KerasFeatureModel.dataset_robustness:
                TensorflowGenerator.X_robustness = x_test[0:TensorflowGenerator.default_robustness_set_size]
                TensorflowGenerator.Y_robustness = y_test[0:TensorflowGenerator.default_robustness_set_size]

            TensorflowGenerator.dataset = dataset

    @staticmethod
    def export_png(model, path):
        if not TensorflowGenerator.model_graph_export:
            return

        from keras.utils import plot_model
        try:
            print("saving model png to {}.png".format(path))
            plot_model(model, to_file='{}.png'.format(path))
        except Exception as e:
            print("error export model image: {}".format(e))
            TensorflowGenerator.model_graph_export = False

    def print(self, include_summary=True, invalid_params=True, export_png=True):
        model = self.model.model
        if include_summary:
            model.summary()
            
        if invalid_params:
            missing_params = self.model.get_custom_parameters()
            for name,(node, params) in missing_params.items():
                print("{0}:{1}".format(name, params))
        
        if TensorflowGenerator.model_graph and export_png:
            TensorflowGenerator.export_png(model, TensorflowGenerator.model_graph)
            
    def load_products(self, product):
        def build_rec(node, level=0):
            #print("-"*level + node.get("label"))
            for child in node.get("children"):
                build_rec(child, level+1)

        build_rec(product)
        

#f = open("./lenet5.json", "r")

#product = json.loads(f.read())

#tensorflow = TensorflowGenerator(product)

