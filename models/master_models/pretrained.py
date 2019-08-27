import numpy as np
from importlib import import_module
import tensorflow as tf
import tensorflow.keras.applications as tfapps
from tensorflow.keras import Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten

def pretrained_model(which_model="VGG16", input_dim=(16, 16, 3), output_depth=None):
    """ Setup an instance of any model available in tensorflow.keras with 
        new input dimensions and a custom depth at which to get the output. 
        The goal of this model is to extract features from the new input using 
        pretrained weights from imagenet. 

        param which_model:  string, which model to use. Defaults to VGG16
        param input_dim:    dimensions of input, (16,16,3) is default for scintillator
        param output_depth: At which layer to get the output from. If not provided
                            returns full model (default) up to, but not including the
                            dense layers. ("include_top=False")
                            Same happens if provided depth is larger than number
                            of layers.
    """

    # Dictionary of possible models and their import statements
    available_models = {
            "DenseNet121":".densenet",
            "DenseNet169":".densenet",
            "DenseNet201":".densenet",
            "InceptionResNetV2":".inception_resnet_v2",
            "InceptionV3":".inception_v3",
            "MobileNet":".mobilenet",
            "MobileNetV2":".mobilenet_v2",
            "NASNetLarge":".nasnet",
            "NASNetMobile":".nasnet",
            "ResNet50":".resnet50",
            "VGG16":".vgg16",
            "VGG19":".vgg19",
            "Xception":".xception",
            }

    # Check if input model is valid
    if which_model not in available_models.keys():
        print("Model not valid. Possible models are:")
        print(available_models)
        exit(1)

    # import correct module
    base_name = "tensorflow.keras.applications"
    module_name = base_name + available_models[which_model]
    module = import_module(module_name)

    # Load the actual function which lets us create a new instance of a model
    pretrained = getattr(module, which_model)(include_top=False, weights='imagenet')

    # Create new input layer
    input_layer = Input(shape=input_dim)

    # Add input layer and desired amount of pretrained layers to new model
    model = Sequential()
    model.add(input_layer)
    if output_depth is not None:
        for i in range(1, output_depth):
            try:
                model.add(pretrained.layers[i])
            except IndexError:
                # Break if output_depth provided is deeper than model.
                break
            except ValueError:
                # Break if sequential model doesn't support the layer.
                break
    else:
        for i in range(1, len(pretrained.layers)):
            try:
                model.add(pretrained.layers[i])
            except IndexError:
                # Break if output_depth provided is deeper than model.
                break
            except ValueError:
                # Break if sequential model doesn't support the layer.
                break

    # Flatten layer to prep for inputting to dense
    model.add(Flatten())

    return model

if __name__ == "__main__":
    pretrained_model("DenseNet121")
