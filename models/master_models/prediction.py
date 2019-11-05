import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D

def position_cnn(input_shape=(16,16,1)):
    """ Set up a sequential model for prediction of positions.
    """
    model = Sequential()

    # Add layers
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4))
    model.add(Activation('linear'))

    return model

def position_dense(input_shape=(256, 1)):
    """ Set up a sequential model for prediction of positions
    using a fully-connected network.
    """
    model = Sequential()

    # Add layers
    model.add(Dense(512), input_shape=input_shape)
    model.add(Activation('relu'))
    model.add(Dense(512), input_shape=input_shape)
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4))
    model.add(Activation('linear'))

    return model
