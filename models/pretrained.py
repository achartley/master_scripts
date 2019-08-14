import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.engine.input_layer import Input
from keras.models import Model
from keras.layers import Dense,Flatten
from keras import backend as K

def vgg_model(input_dim=(16, 16, 3), output_depth=0):
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
    input_layer = Input(shape=input_dim)
    vgg = VGG16(include_top=False, input_tensor=input_layer, weights='imagenet')

    # Set output_layer to full model, or to the desired depth if provided.
    if output_depth > 0:
        output_layer = Flatten()(vgg.layers[-1].output)
    else:
        output_layer = Flatten()(vgg.layers[output_depth].output)

    # Keras infers the tensors between input and output layers.
    model = Model(inputs=input_layer, outputs=output_layer) 

    return model
