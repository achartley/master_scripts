import numpy as np
import json
from importlib import import_module
import tensorflow as tf
import tensorflow.keras.applications as tfapps
from tensorflow.keras import Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

def project_model(compiled=True):
    """ Setup an instance of the model made by Harrison Labollita in a previous
        experiment with scintillator data.

        :param compiled:    bool, if true compiled with same settings as in the
                            project. If false, compilation must be done outside
                            this function.

        :return:    Instance of the Sequential model.
    """

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation = 'relu', input_shape= (16,16,1)))
    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(2, activation = 'softmax'))
    if compiled:
        model.compile(
                loss = 'categorical_crossentropy',
                optimizer = 'adadelta',
                metrics = ['accuracy']
                )

    return model


if __name__ == "__main__":
    model = project_model(compiled=False)
    model_config = model.to_json()
    print(json.dumps(json.loads(model_config), indent=2))
