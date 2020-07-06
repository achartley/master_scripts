from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from master_scripts.data_functions import get_tf_device, get_git_root
import tensorflow as tf
import json
import warnings
import hashlib
# Set warning options to get a bit cleaner output when running
warnings.filterwarnings('ignore', category=FutureWarning)


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

    def __init__(self, experiment_type, model=None, gpu_max_load=20,
                 config=None
                 ):
        """ The config should contain the parameters needed for
        model.fit. If no config is given, default values are used.

        param experiment_type: 'classification' or 'prediction'
        param experiment_name: optional name to include in output title
        param model: the model that is to be trained and evaluated
        """

        self.experiment_type = experiment_type
        self.model = model
        self.experiment_id = None

        self.history = None
        self.history_kfold = None

        # set gpu/cpu device defaults
        self.tf_device = get_tf_device(gpu_max_load)

        # Set default config and append or replace with provided config.
        self.config = None
        self.set_config(config)
        self.set_experiment_id()

    def set_experiment_id(self):
        """ Creates a unique hash from the config parameters.
        """
        # yyyymmddhhmmssffffff where f is microseconds
        id_string = datetime.today().strftime('%Y%m%d%H%M%S%f')
        # Add stringified config values
        for k in self.config.keys():
            for v in self.config[k]:
                id_string += str(v)

        # Hash the full id_string and set id to first 12 elements
        m = hashlib.md5()
        m.update(id_string.encode('utf-8'))
        self.experiment_id = m.hexdigest()[:12]

    def set_config(self, config):
        """ Set default config for model.fit and additional kfold cross-
        validation arguments. If a config is provided to the function,
        replace given keys with options.
        """
        # Get github repo root path
        rpath = get_git_root()
        self.config = {
            'fit_args': {
                'batch_size': None,
                'epochs': 1,
                'verbose': 1,
                'callbacks': None,
                'validation_split': 0.,
                'validation_data': None,
                'shuffle': True,
                'class_weight': None,
                'sample_weight': None,
                'initial_epoch': 0,
                'steps_per_epoch': None,
                'validation_steps': None,
                'validation_batch_size': None,
                'validation_freq': 1,
                'max_queue_size': 10,
                'workers': 1,
                'use_multiprocessing': False
            },
            'kfold_args': {
                'n_splits': 5,
                'shuffle': False,
                'random_state': None,
            },
            'path_args': {
                'repo_root': rpath,
                'models': rpath + 'models/',
                'figures': rpath + 'figures/',
                'experiments': rpath + 'experiments/',
                'results': rpath + 'results',
            },
        }
        if config is not None:
            for major_key in config.keys():
                for k, v in config[major_key].items():
                    self.config[major_key][k] = v

    def run(self, x_train, y_train, x_val, y_val):
        """ Single training run of the given model. It is assumed that the
        input data is preprocessed, normalized, and good to go.
        """
        self.config['fit_args']['validation_data'] = (x_val, y_val)
        # Determine tensorflow device if not provided
        if self.tf_device is None:
            self.tf_device = get_tf_device(self.gpu_max_load)

        with tf.device(self.tf_device):
            self.history = self.model.fit(
                x=x_train,
                y=y_train,
                **self.config['fit_args'],
            )

    def run_kfold(self, x, y):
        """ Train the model using kfold cross-validation.
        """

        # StratifiedKFold doesn't take one-hot
        y = y.argmax(axis=-1)

        # Store accuracy for each fold for all models
        results = {}

        # Create KFold data generator
        skf = StratifiedKFold(**self.config['kfold_args'])

        # Run k-fold cross-validation
        fold = 0  # Track which fold
        for train_idx, val_idx in skf.split(x, y):
            self.config['fit_args']['validation_data'] = (x[val_idx],
                                                          y[val_idx])

            # Train model
            history = self.model.fit(
                x[train_idx],
                y[train_idx],
                **self.config['fit_args'],
            )
            # Store the accuracy
            results[fold] = history
            fold += 1
        self.history_kfold = results

    def output_experiment(self):
        """ Output a structured json file containing all model configuration
        and parameters, as well as model evaulation metrics and results.
        """
        output = {}
        output['loss'] = self.model.loss
        output['metrics'] = self.model.metrics_names
        output['optimizer'] = self.model.optimizer.get_config
        output['model'] = self.model.get_config()
        output['experiment_config'] = self.config
        output['experiment_type'] = self.experiment_type
        output['experiment_id'] = self.experiment_id
        if self.history_kfold:
            output['history'] = self.history_kfold
        else:
            output['history'] = self.history

        fpath = self.config['path_args']['experiments'] + \
            self.experiment_id + ".json"

        with open(fpath, 'w') as fp:
            json.dump(output, fp, indent=2)


if __name__ == "__main__":

    test = Experiment(
        experiment_type='classification',

    )
