from keras import models, layers
import keras
from keras.datasets import mnist, cifar10
from keras import backend as K
import tensorflow as tf
import time

class AlexNet(models.Sequential):
    optimizers = []
    losss = ['categorical_crossentropy']

    def __init__(self, input_shape, num_classes):
        super().__init__()

            

        self.add(layers.Conv2D(filters=96, input_shape=input_shape, kernel_size=(11,11),strides=(4,4), padding='valid'))
        self.add(layers.Activation('relu'))
        # Pooling
        self.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        # Batch Normalisation before passing it to the next layer
        self.add(layers.BatchNormalization())

        # 2nd Convolutional Layer
        self.add(layers.Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
        self.add(layers.Activation('relu'))
        # Pooling
        self.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        # Batch Normalisation
        self.add(layers.BatchNormalization())

        # 3rd Convolutional Layer
        self.add(layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
        self.add(layers.Activation('relu'))
        # Batch Normalisation
        self.add(layers.BatchNormalization())

        # 4th Convolutional Layer
        self.add(layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
        self.add(layers.Activation('relu'))
        # Batch Normalisation
        self.add(layers.BatchNormalization())

        # 5th Convolutional Layer
        self.add(layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
        self.add(layers.Activation('relu'))
        # Pooling
        self.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        # Batch Normalisation
        self.add(layers.BatchNormalization())

        # Passing it to a dense layer
        self.add(layers.Flatten())
        # 1st Dense Layer
        self.add(layers.Dense(4096, input_shape=input_shape))
        self.add(layers.Activation('relu'))
        # Add Dropout to prevent overfitting
        self.add(layers.Dropout(0.4))
        # Batch Normalisation
        self.add(layers.BatchNormalization())

        # 2nd Dense Layer
        self.add(layers.Dense(4096))
        self.add(layers.Activation('relu'))
        # Add Dropout
        self.add(layers.Dropout(0.4))
        # Batch Normalisation
        self.add(layers.BatchNormalization())

        # 3rd Dense Layer
        self.add(layers.Dense(1000))
        self.add(layers.Activation('relu'))
        # Add Dropout
        self.add(layers.Dropout(0.4))
        # Batch Normalisation
        self.add(layers.BatchNormalization())
        self.add(layers.Dense(num_classes, activation='softmax'))

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

    model = AlexNet(input_shape, num_classes)

    begin_training = time.time()
    model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('model params:', model.count_params())

    #print("flops", get_flops(model))


if __name__ == "__main__":
    run("cifar", 12)