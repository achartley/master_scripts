import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from master_scripts.data_functions import get_tf_device
from master_scripts.models_classification import project_model


class Experiment:
    """ Class that handles a full training run of one model.
    The class takes a defined, compiled model, runs training and logs
    model parameters and results to file. The model is stored with an id
    matching that of the output configuration.

    The idea of one experiment is:
    - before experiment run
        Data is preprocessed
        Model is defined and compiled with desired optimizer and hyperparams
        Data is split into training and validation based on whether it's
        one training run or if it's kfold.
    - model is trained with given parameters
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

    def __init__(self, experiment_type, experiment_name=None, model=None):
        """ The config should contain the parameters needed for
        model.fit. If no config is given, default values are used.

        param experiment_type: 'classification' or 'prediction'
        param experiment_name: optional name to include in output title
        param model: the model that is to be trained and evaluated
        """

        self.model = model
        self.history = None
        self.experiment_type = experiment_type
        # set experiment id, format from datetime today
        # yyyymmddhhmmssffffff where f is microseconds
        self.experiment_id = datetime.today().strftime('%Y%m%d%H%M%S%f')
        self.experiment_name = experiment_name

        # set gpu/cpu device
        self.gpu_max_load = 20
        self.tf_device = None

        # minor default config
        self.config = {
            'batch_size': 32,
            'epochs': 20,
            'verbose': 2,
            'callbacks': []
        }

    def run(self, x_train, y_train, x_val, y_val,
            TF_DEVICE=None,
            ):
        """ Single training run of the given model. It is assumed that the
        input data is preprocessed, normalized, and good to go.
        """
        # Determine tensorflow device if not provided
        if self.tf_device is None:
            self.tf_device = get_tf_device(self.gpu_max_load)

        # PATH variables
        DATA_PATH = "../../data/simulated/"
        OUTPUT_PATH = "../../data/output/"
        MODEL_PATH = OUTPUT_PATH + "models/"

        with tf.device(self.tf_device):
            self.history = self.model.fit(
                x_train,
                y_train,
                batch_size=self.config['batch_size'],
                epochs=self.config['epochs'],
                validation_data=(x_val, y_val),
                verbose=self.config['verbose'],
                callbacks=self.config['callbacks']
            )

    def run_kfold(self, x, y, TF_DEVICE=None):
        """ Train the model using kfold cross-validation.
        """
        # Determine tensorflow device if not provided
        if self.tf_device is None:
            self.tf_device = get_tf_device(self.gpu_max_load)

        if 'n_folds' in self.config:
            n_folds = self.config['n_folds']
        else:
            n_folds = 5

        # Params for k-fold cross-validation
        k_shuffle = True

        # StratifiedKFold doesn't take one-hot
        y = y.argmax(axis=-1)

        # Store accuracy for each fold for all models
        k_fold_results = {}

        # Create KFold data generator
        skf = StratifiedKFold(n_splits=n_folds, shuffle=k_shuffle)

        # Run k-fold cross-validation
        fold = 0
        for train_idx, val_idx in skf.split(x, y):

            # Train model
            history = self.model.fit(
                x[train_idx],
                y[train_idx],
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size'],
                validation_data=(x[val_idx], y[val_idx]),
                callbacks=self.config['callbacks']
            )

            # Store the accuracy
            k_fold_results[fold] = history

        raise NotImplementedError

    def output_experiment(self):
        """ Output a structured json file containing all model configuration
        and parameters, as well as model evaulation metrics and results.
        """
        raise NotImplementedError
