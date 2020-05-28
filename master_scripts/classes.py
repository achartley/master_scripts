# Ignore FutureWarnings
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import json
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold

from master_scripts.data_functions import *

class Experiment:
    """ Class that handles a full training run of one model. 
    The class takes a defined, compiled model, runs training and logs
    model parameters and results to file. The model is stored with an id
    matching that of the output configuration.

    The idea of one experiment is:
    - before experiment run
        Data is preprocessed
        Model is defined and compiled with desired optimizer and hyperparams
    - during run
        Data is split into training and validation based on whether it's
        one training run or if it's kfold.
        The model is trained.
    - after training
        Model is saved so that it can be loaded and used for prediction or
        classification later.
        Model configuration (with hyperparameters etc) is stored in a
        experiment file with a unique ID in such a way that the
        model can be rebuilt from that file.
        Any metrics, and the history object of the training run are also
        serialized and stored in this experiment file for post-experiment
        inspection.
    """

    def __init__(self, model=None, config=None):
        """ The config should contain the parameters needed for
        model.fit. If no config is given, default values are used.
        """
        self.model = model
        self.config = config
        # set experiment id, format from datetime today
        # yyyymmddhhmmss
        self.experiment_id = datetime.today().strftime('%Y%m%d%H%M%S')
        self.experiment_name = name

        # set gpu/cpu device
        self.gpu_max_load = 20
        self.tf_device = None
    

    def load_config(self, config_path):
        """ The config is json-formatted.
        See ../config/ for example and/or template
        """
        with open(config_path, "r") as fp:
            config = json.load(config_path)

    def run(x_train, y_train, x_val, y_val, MAX_LOAD=None, TF_DEVICE=None):
        """ Single training run of the given model. It is assumed that the
        input data is preprocessed, normalized and good to go.
        """
        # Determine tensorflow device
        if self.tf_device is None:
            self.tf_device = get_tf_device(self.gpu_max_load)  

        # PATH variables
        DATA_PATH = "../../data/simulated/"
        OUTPUT_PATH = "../../data/output/"
        MODEL_PATH = OUTPUT_PATH + "models/"

        with tf.device(DEVICE):
            # Callbacks
            cb_save = tf.keras.callbacks.ModelCheckpoint(
                    filepath=fpath, 
                    monitor='val_loss', 
                    save_best_only=True,
                    mode='min'
                    )
            cb_earlystopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=2,
                    verbose=1,
                    restore_best_weights=True,
                    )

            cb_r2 = R2ScoreCallback(val_data)
            # Parameters for the model
            batch_size = 32
            epochs = 20
            history = model.fit(
                    x_train,
                    y_train,
                    batch_size=self.config["batch_size"],
                    epochs=self.config["epochs"],
                    validation_data=(x_val, y_val),
                    verbose=self.config["verbose"],
                    callbacks=[cb_r2, cb_earlystopping, cb_save]
                    )
            test_predictions = model.predict(normalize_image_data(images[test_idx]))
            test_r2 = cb_r2.r2_score(test_predictions, positions[test_idx, :2])
            print("test_r2:", test_r2)
        raise NotImplemented

    def run_kfold(x, y, MAX_LOAD=None, TF_DEVICE=None ):
        """ Train the model using kfold cross-validation.
        """
        raise NotImplemented

    def output_experiment():
        """ Output a structured json file containing all model configuration
        and parameters, as well as model evaulation metrics and results.
        """
        raise NotImplemented


