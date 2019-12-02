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
                     input_shape=input_shape, activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(1600, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(4))
    model.add(Activation('linear'))

    return model

def position_energy_cnn(input_shape=(16,16,1)):
    """ Set up a sequential model for prediction of positions.
    """
    model = Sequential()

    # Add layers
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=input_shape, activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(6))
    model.add(Activation('linear'))

    return model
def position_dense():
    """ Set up a sequential model for prediction of positions
    using a fully-connected network.
    """
    model = Sequential()

    # Add layers
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4))
    model.add(Activation('linear'))

    return model

def position_project():
    """ The same model as used for classification, but without
    the top dense layers.
    """

    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape= (16,16,1), activation = 'relu'))
    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(4, activation = 'linear'))

    return model
