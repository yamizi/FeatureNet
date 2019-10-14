from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import Callback, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import os, time, math
import keras
import numpy as np
import tensorflow as tf



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
    print('Learning rate: ', lr)
    return lr

def train_model(model, x_train, y_train, x_test, y_test, epochs, batch_size, scheduler=False, data_augmentation=False, model_path=None):

    
    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                cooldown=0,
                                patience=5,
                                min_lr=0.5e-6)


    early_stopping = EarlyStopping(monitor='val_acc', mode='max', min_delta=0.01, patience=20)
    #timed = TimedStopping(model,None, 6000)

    callbacks = []#, lr_reducer, lr_scheduler]

    if scheduler:
        callbacks = [lr_reducer, lr_scheduler, early_stopping]        

    if not model_path:
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        model_name = 'temp_model.h5'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        model_path = os.path.join(save_dir, model_name)

    mc = ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True)
    callbacks.append(mc)
    history = None

    try:

        # Run training, with or without data augmentation.
        if not data_augmentation:
            print('Not using data augmentation.')
            history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    shuffle=True,
                    callbacks=callbacks,
                    verbose=2)
        else:
            print('Using real-time data augmentation.')
            # This will do preprocessing and realtime data augmentation:
            datagen = ImageDataGenerator(
                # set input mean to 0 over the dataset
                featurewise_center=False,
                # set each sample mean to 0
                samplewise_center=False,
                # divide inputs by std of dataset
                featurewise_std_normalization=False,
                # divide each input by its std
                samplewise_std_normalization=False,
                # apply ZCA whitening
                zca_whitening=False,
                # epsilon for ZCA whitening
                zca_epsilon=1e-06,
                # randomly rotate images in the range (deg 0 to 180)
                rotation_range=0,
                # randomly shift images horizontally
                width_shift_range=0.1,
                # randomly shift images vertically
                height_shift_range=0.1,
                # set range for random shear
                shear_range=0.,
                # set range for random zoom
                zoom_range=0.,
                # set range for random channel shifts
                channel_shift_range=0.,
                # set mode for filling points outside the input boundaries
                fill_mode='nearest',
                # value used for fill_mode = "constant"
                cval=0.,
                # randomly flip images
                horizontal_flip=True,
                # randomly flip images
                vertical_flip=False,
                # set rescaling factor (applied before any other transformation)
                rescale=None,
                # set function that will be applied on each input
                preprocessing_function=None,
                # image data format, either "channels_first" or "channels_last"
                data_format=None,
                # fraction of images reserved for validation (strictly between 0 and 1)
                validation_split=0.0)

            # Compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied).
            datagen.fit(x_train)

            steps_per_epoch= math.ceil(len(x_train) / batch_size)
            # Fit the model on the batches generated by datagen.flow().
            history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                steps_per_epoch=steps_per_epoch,
                                validation_data=(x_test, y_test),
                                epochs=epochs, verbose=2, workers=4,
                                callbacks=callbacks)

        model = keras.models.load_model(model_path)
        os.remove(model_path)

    except tf.errors.ResourceExhaustedError as e:
        print("model exhausted memory while training, error {}".format(e))
        model = None
    return model, history