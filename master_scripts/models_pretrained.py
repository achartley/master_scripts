from importlib import import_module
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten


def pretrained_model(which_model="VGG16", input_dim=(16, 16, 3),
                     output_depth=None):
    """ Setup an instance of any model available in tensorflow.keras with
        new input dimensions and a custom depth at which to get the output.
        The goal of this model is to extract features from the new input using
        pretrained weights from imagenet.

        param which_model:  string, which model to use. Defaults to VGG16
        param input_dim:    dimensions of input, default to (16, 16, 3)
        param output_depth: At which layer to get the output from. If not
            provided returns full model (default) up to, but not including the
            dense layers ("include_top=False") Same happens if provided depth
            is larger than number of layers.
    """

    # Dictionary of possible models and their import statements
    available_models = {
        "DenseNet121": ".densenet",
        "DenseNet169": ".densenet",
        "DenseNet201": ".densenet",
        "InceptionResNetV2": ".inception_resnet_v2",
        "InceptionV3": ".inception_v3",
        "MobileNet": ".mobilenet",
        "MobileNetV2": ".mobilenet_v2",
        "NASNetLarge": ".nasnet",
        "NASNetMobile": ".nasnet",
        "ResNet50": ".resnet50",
        "VGG16": ".vgg16",
        "VGG19": ".vgg19",
        "Xception": ".xception",
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
    pretrained = getattr(module, which_model)(
        include_top=False, weights='imagenet')

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
                print("ValueError met, breaking the layer addition loop")
                break

    # Flatten layer to prep for inputting to dense
    model.add(Flatten())

    return model


def pretrained_vgg16(input_dim=(16, 16, 3)):
    """ Setup an instance of vgg16 pretrained on imagenet.

        :param input_dim:    dimensions of input, (16,16,3) is default for
                            scintillator
        :returns model: Non-compiled Sequential model with pretrained layers.
    """

    # import correct module
    module_name = "tensorflow.keras.applications.vgg16"
    module = import_module(module_name)

    # Load the actual function which lets us create a new instance of a model
    pretrained = getattr(module, "VGG16")(include_top=True, weights='imagenet')

    # Create new input layer
    input_layer = Input(shape=input_dim)

    # Add input layer and all pretrained layers except final softmax layer
    model = Sequential()
    model.add(input_layer)
    for i in range(1, len(pretrained.layers)):
        try:
            model.add(pretrained.layers[i])
        except IndexError:
            # Break if output_depth provided is deeper than model.
            break
        except ValueError as err:
            # Break if sequential model doesn't support the layer.
            print("ValueError:", err)
            break

    return model


def pretrained_resnet50(input_dim=(16, 16, 3)):
    """ Setup an instance of resnet50 pretrained on imagenet

        param input_dim:    dimensions of input, (16,16,3) is default for
                            scintillator
    """

    # import correct module
    module_name = "tensorflow.keras.applications.resnet50"
    module = import_module(module_name)

    # Load the actual function which lets us create a new instance of a model
    pretrained = getattr(module, "ResNet50")(
        include_top=True, weights='imagenet')

    # Create new input layer
    input_layer = Input(shape=input_dim)

    # Add input layer and all pretrained layers except final softmax layer
    model = Sequential()
    model.add(input_layer)
    for i in range(0, len(pretrained.layers) - 1):
        try:
            model.add(pretrained.layers[i])
        except IndexError:
            # Break if output_depth provided is deeper than model.
            break
        except ValueError as err:
            # Break if sequential model doesn't support the layer.
            print("ValueError:", err)
            break

    return model


if __name__ == "__main__":
    pretrained_model("DenseNet121")
