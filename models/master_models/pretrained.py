import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten

def vgg_model(input_dim=(16, 16, 3), output_depth=9):
    """ Setup an instance of the VGG16 model with new input dimensions
        and a custom depth at which to get the output. The goal of this
        model is to extract features from the new input using VGG's existing
        weights. The features can be stored on their own and the be run
        through a dense network for classification.

        param input_dim:    dimensions of input, (16,16,3) is default for scintillator
        param output_depth: At which layer to get the output from. 0 corresponds
                            to the full model up to, but not including the
                            dense layers. ("include_top=False")
    """
    # Create new input layer
    input_layer = Input(shape=input_dim)

    # Load vgg16
    vgg = VGG16(include_top=False, weights='imagenet')

    # Add input layer and desired amount of vgg16 layers to new model
    model = Sequential()
    model.add(input_layer)
    for i in range(1, output_depth):
        model.add(vgg.layers[i])

    # Flatten layer to prep for inputting to dense
    model.add(Flatten())

    return model
