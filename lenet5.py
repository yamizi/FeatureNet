from keras import models, layers
import keras
from keras.datasets import mnist, cifar10
from keras import backend as K
import tensorflow as tf
import time

class LeNet(models.Sequential):
    
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=input_shape, padding="same"))
        self.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
        self.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        self.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.add(layers.Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        self.add(layers.Flatten())
        self.add(layers.Dense(84, activation='tanh'))
        self.add(layers.Dense(num_classes, activation='softmax'))

        self.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer="sgd")


def run(dataset="mnist", epochs=12, index=0):
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

    model = LeNet(input_shape, num_classes)

    begin_training = time.time()
    history = model.fit(x_train, y_train,
        batch_size=batch_size,
        validation_data=(x_test, y_test), 
        epochs=epochs,
        verbose=1)

    training_time = time.time() - begin_training

    h = (history.history['acc'], history.history['val_acc'])
    h = "{accuracy}|{validation_accuracy}".format(accuracy="#".join(map(str, h[0])), validation_accuracy="#".join(map(str, h[1])))
               

    score = model.evaluate(x_test, y_test, verbose=0)
    #print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    #print('model params:', model.count_params())


    f2 = open("report_lenet5_cifar_validation.txt","a")
    f2.write("lenet5_raw{0}: {1} {2} - - - {3}".format(index, score[1], model.count_params(), h))
    f2.close()


for i in range(10):
    print("sprint {}".format(i))
    run("cifar", 150, i)