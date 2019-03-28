from keras.models import Model
from keras.layers import Input, Activation, Concatenate
from keras.layers import Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D

import keras
from keras.datasets import mnist, cifar10
from keras import backend as K
import tensorflow as tf
import time

class SqueezeNet(Model):
    optimizers = []
    losss = ['categorical_crossentropy']

    def __init__(self, inputs, nb_classes):
        
        input_img = Input(shape=inputs)
        conv1 = Convolution2D(
            96, (7, 7), activation='relu', kernel_initializer='glorot_uniform',
            strides=(2, 2), padding='same', name='conv1',
            data_format="channels_first")(input_img)
        maxpool1 = MaxPooling2D(
            pool_size=(1, 1), strides=(2, 2), name='maxpool1',
            data_format="channels_first")(conv1)
        fire2_squeeze = Convolution2D(
            16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire2_squeeze',
            data_format="channels_first")(maxpool1)
        fire2_expand1 = Convolution2D(
            64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire2_expand1',
            data_format="channels_first")(fire2_squeeze)
        fire2_expand2 = Convolution2D(
            64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire2_expand2',
            data_format="channels_first")(fire2_squeeze)
        merge2 = Concatenate(axis=1)([fire2_expand1, fire2_expand2])

        fire3_squeeze = Convolution2D(
            16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire3_squeeze',
            data_format="channels_first")(merge2)
        fire3_expand1 = Convolution2D(
            64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire3_expand1',
            data_format="channels_first")(fire3_squeeze)
        fire3_expand2 = Convolution2D(
            64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire3_expand2',
            data_format="channels_first")(fire3_squeeze)
        merge3 = Concatenate(axis=1)([fire3_expand1, fire3_expand2])

        fire4_squeeze = Convolution2D(
            32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire4_squeeze',
            data_format="channels_first")(merge3)
        fire4_expand1 = Convolution2D(
            128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire4_expand1',
            data_format="channels_first")(fire4_squeeze)
        fire4_expand2 = Convolution2D(
            128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire4_expand2',
            data_format="channels_first")(fire4_squeeze)
        merge4 = Concatenate(axis=1)([fire4_expand1, fire4_expand2])
        maxpool4 = MaxPooling2D(
            pool_size=(1, 1), strides=(2, 2), name='maxpool4',
            data_format="channels_first")(merge4)

        fire5_squeeze = Convolution2D(
            32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire5_squeeze',
            data_format="channels_first")(maxpool4)
        fire5_expand1 = Convolution2D(
            128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire5_expand1',
            data_format="channels_first")(fire5_squeeze)
        fire5_expand2 = Convolution2D(
            128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire5_expand2',
            data_format="channels_first")(fire5_squeeze)
        merge5 = Concatenate(axis=1)([fire5_expand1, fire5_expand2])

        fire6_squeeze = Convolution2D(
            48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire6_squeeze',
            data_format="channels_first")(merge5)
        fire6_expand1 = Convolution2D(
            192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire6_expand1',
            data_format="channels_first")(fire6_squeeze)
        fire6_expand2 = Convolution2D(
            192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire6_expand2',
            data_format="channels_first")(fire6_squeeze)
        merge6 = Concatenate(axis=1)([fire6_expand1, fire6_expand2])

        fire7_squeeze = Convolution2D(
            48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire7_squeeze',
            data_format="channels_first")(merge6)
        fire7_expand1 = Convolution2D(
            192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire7_expand1',
            data_format="channels_first")(fire7_squeeze)
        fire7_expand2 = Convolution2D(
            192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire7_expand2',
            data_format="channels_first")(fire7_squeeze)
        merge7 = Concatenate(axis=1)([fire7_expand1, fire7_expand2])

        fire8_squeeze = Convolution2D(
            64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire8_squeeze',
            data_format="channels_first")(merge7)
        fire8_expand1 = Convolution2D(
            256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire8_expand1',
            data_format="channels_first")(fire8_squeeze)
        fire8_expand2 = Convolution2D(
            256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire8_expand2',
            data_format="channels_first")(fire8_squeeze)
        merge8 = Concatenate(axis=1)([fire8_expand1, fire8_expand2])

        maxpool8 = MaxPooling2D(
            pool_size=(1, 1), strides=(2, 2), name='maxpool8',
            data_format="channels_first")(merge8)
        fire9_squeeze = Convolution2D(
            64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire9_squeeze',
            data_format="channels_first")(maxpool8)
        fire9_expand1 = Convolution2D(
            256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire9_expand1',
            data_format="channels_first")(fire9_squeeze)
        fire9_expand2 = Convolution2D(
            256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='fire9_expand2',
            data_format="channels_first")(fire9_squeeze)
        merge9 = Concatenate(axis=1)([fire9_expand1, fire9_expand2])

        fire9_dropout = Dropout(0.5, name='fire9_dropout')(merge9)
        conv10 = Convolution2D(
            nb_classes, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='valid', name='conv10',
            data_format="channels_first")(fire9_dropout)

        global_avgpool10 = GlobalAveragePooling2D(data_format='channels_first')(conv10)
        softmax = Activation("softmax", name='softmax')(global_avgpool10)

        super().__init__(inputs=input_img, outputs=softmax)
        self.compile(loss=self.losss[0], metrics=['accuracy'], optimizer=self.optimizers[0] if len(self.optimizers) else "sgd")


def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops


def run(dataset="mnist", epochs=12):
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

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    model = SqueezeNet(input_shape, num_classes)

    history = model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    h = (history.history['acc'], history.history['val_acc'])
    h = "{accuracy}|{validation_accuracy}".format(accuracy="#".join(map(str, h[0])), validation_accuracy="#".join(map(str, h[1])))
    
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('model params:', model.count_params())

    f2 = open("squeezenet_cifar_validation.txt","a")
    index = 0
    f2.write("{0}: {1} {2} - - - {3}".format(index, score[1], model.count_params(), h))
    f2.close()


    #print("flops", get_flops(model))

run("cifar", 300)

#Test accuracy: 0.1796
#model params: 876970