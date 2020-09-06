from datetime import datetime
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (matthews_corrcoef, f1_score, confusion_matrix,
                             roc_auc_score, accuracy_score, r2_score,
                             mean_squared_error, mean_absolute_error)
from master_scripts.data_functions import get_git_root
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

    def __init__(self, model=None, config=None, model_type=None,
                 experiment_name=None):
        """ The config should contain the parameters needed for
        model.fit. If no config is given, default values are used.

        param model: the model that is to be trained and evaluated
        param model_type: 'classification' or 'regression'
        param experiment_name: optional name to include in output
        """

        self.model = model
        self.model_type = model_type
        self.experiment_name = experiment_name
        self.id = None

        self.history = None
        self.metrics = {}
        self.indices = {}

        self.history_kfold = None
        self.metrics_kfold = {}

        # Set default config and append or replace with provided config.
        self.config = None
        self.set_config(config)
        self.set_experiment_id()

    def set_config(self, config):
        """ Set default config for model.fit and additional kfold cross-
        validation arguments. If a config is provided to the function,
        replace given keys with the ones provided, and add new keys present.
        """
        # Get github repo root path
        rpath = get_git_root()
        self.config = {
            'fit_args': {
                'batch_size': None,
                'epochs': 1,
            },
            'kfold_args': {
                'n_splits': 5,
                'shuffle': False,
            },
            'path_args': {
                'repo_root': rpath,
                'models': rpath + 'models/',
                'figures': rpath + 'figures/',
                'experiments': rpath + 'experiments/',
                'results': rpath + 'results',
                'model_config': rpath + 'experiments/model_config/',
            },
            'random_seed': None,
        }
        if config is not None:
            for major_key in config.keys():
                # Treat the value as dict, save directly if not dict
                try:
                    for k, v in config[major_key].items():
                        self.config[major_key][k] = v
                except AttributeError:
                    self.config[major_key] = config[major_key]
                except KeyError:
                    self.config[major_key] = config[major_key]

    def set_experiment_id(self):
        """ Creates a unique hash from the config parameters.
        """
        # yyyymmddhhmmssffffff where f is microseconds
        id_string = datetime.today().strftime('%Y%m%d%H%M%S%f')
        # Add stringified config values
        for k in self.config.keys():
            try:
                for v in self.config[k].values():
                    id_string += str(v)
            except AttributeError:
                id_string += str(self.config[k])

        # Hash the full id_string and set id to first 12 elements
        m = hashlib.md5()
        m.update(id_string.encode('utf-8'))
        self.id = m.hexdigest()[:12]

    def load_model(self, experiment_id):
        """ Load a model from a given experiment.

        param experiment_id: 12-character hash id from some previously
            executed experiment
        """
        model_dir = self.config['path_args']['model_config']
        self.model = tf.keras.model_from_json(model_dir + experiment_id)

    def run(self, x, y):
        """ Single training run of the given model. It is assumed that the
        input data is preprocessed, normalized, and good to go.

        :param x:   input data (batch_size, width, height, channels)
        :param y:   labels
        """

        # Set indices for train and validation
        x_idx = np.arange(x.shape[0])
        train_idx, val_idx = train_test_split(
            x_idx,
            random_state=self.config['random_seed']
        )

        # Train the model
        self.history = self.model.fit(
            x=x[train_idx],
            y=y[train_idx],
            validation_data=(x[val_idx], y[val_idx]),
            **self.config['fit_args'],
        ).history

        # Calculate metrics for the model
        if self.model_type == "classification":
            self.classification_metrics(x[val_idx], y[val_idx])
        elif self.model_type == "regression":
            self.regression_metrics(x[val_idx], y[val_idx])

        # Store indices for training and validation in config output
        # Need conversion to list as numpy arrays aren't json serializable.
        # To make the indices in config more uniform in format, we treat
        # a non-kfold experiment like a 1-fold experiment.
        self.indices['fold_0'] = {
            'train_idx': train_idx.tolist(),
            'val_idx': val_idx.tolist(),
        }

    def run_kfold(self, x, y):
        """ Train the model using kfold cross-validation.
        """

        # StratifiedKFold doesn't take one-hot
        y = y.argmax(axis=-1)

        # Store accuracy for each fold for all models
        results = {}

        # Create KFold data generator
        skf = StratifiedKFold(
            random_state=self.config['random_seed'],
            **self.config['kfold_args']
        )

        # Run k-fold cross-validation
        fold = 0  # Track which fold
        for train_idx, val_idx in skf.split(x, y):

            # Train model
            history = self.model.fit(
                x[train_idx],
                y[train_idx],
                validation_data=(x[val_idx], y[val_idx]),
                ** self.config['fit_args'],
            )
            # Store the accuracy
            results[fold] = history
            # Calculate metrics for the model
            if self.model_type == "classification":
                self.classification_metrics(x[val_idx], y[val_idx], fold)
            elif self.model_type == "regression":
                self.regression_metrics(x[val_idx], y[val_idx], fold)

            # Store train and val indices for the fold
            foldkey = 'fold_' + str(fold)
            self.indices[foldkey] = {
                'train_idx': train_idx.tolist(),
                'val_idx': val_idx.tolist(),
            }

            fold += 1
        self.history_kfold = results

    def regression_metrics(self, x_val, y_val, fold=None):
        """ Calculates regression metrics on the validation data.
        """

        # Get prediction and make class labels based on threshold of 0.5
        y_pred = self.model.predict(x_val)
        metrics = {}

        metrics['r2_score'] = r2_score(y_val, y_pred)
        metrics['mse'] = mean_squared_error(y_val, y_pred)
        metrics['rmse'] = mean_squared_error(y_val, y_pred, squared=False)
        metrics['mae'] = mean_absolute_error(y_val, y_pred)
        if fold:
            foldkey = 'fold_' + str(fold)
            self.metrics_kfold[foldkey] = metrics
        else:
            self.metrics = metrics

    def classification_metrics(self, x_val, y_val, fold=None):
        """ Calculates f1_score, matthews_corrcoef, confusion matrix and
        roc area under curve, accuracy metrics and stores them in the
        metrics attribute.
        The values are calculated based on the validation data.

        Recall that the default positive class for f1_score is 1
        """

        # Get prediction and make class labels based on threshold of 0.5
        y_out = self.model.predict(x_val)
        y_pred = y_out > 0.5
        metrics = {}
        confmat = confusion_matrix(y_val, y_pred)
        print(confmat)

        metrics['accuracy_score'] = accuracy_score(y_val, y_pred)
        metrics['confusion_matrix'] = {
            'TN': int(confmat[0, 0]),
            'FP': int(confmat[0, 1]),
            'FN': int(confmat[1, 0]),
            'TP': int(confmat[1, 1]),
        }
        metrics['f1_score'] = f1_score(y_val, y_pred)
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_val, y_pred)
        metrics['roc_auc_score'] = roc_auc_score(y_val, y_out)

        if fold:
            foldkey = 'fold_' + str(fold)
            self.metrics_kfold[foldkey] = metrics
        else:
            self.metrics = metrics

    def save(self, save_full=False):
        """ Outputs two files:
        - one experiment config with performance metrics and optimizer
        information, fit parameters etc
        - model config, loadable with tf.keras.models.model_from_json.

        :param save_full: Boolean. Save complete model with weights if true.
        """

        # Collect information into a dictionary
        output = {}
        output['loss'] = self.model.loss
        output['optimizer'] = str(self.model.optimizer.get_config())
        output['experiment_config'] = self.config
        output['model_type'] = self.model_type
        output['experiment_name'] = self.experiment_name
        output['experiment_id'] = self.id
        output['indices'] = self.indices
        output['datetime'] = datetime.today().strftime('%Y-%m-%d %H:%M:%S')

        if self.history_kfold:
            output['history'] = self.history_kfold
            output['metrics'] = self.metrics_kfold
        else:
            output['history'] = self.history
            output['metrics'] = self.metrics

        # Write to files
        experiment_fpath = self.config['path_args']['experiments'] + \
            self.id + ".json"

        model_fpath = self.config['path_args']['model_config'] + \
            "model_" + self.id + ".json"

        # Save the model with weights
        if save_full:
            mpath = self.config['path_args']['models'] + self.id + ".h5"
            self.model.save(mpath)

        # Save experiment
        with open(experiment_fpath, 'w') as fp:
            json.dump(output, fp)

        # Save model config
        with open(model_fpath, 'w') as fp:
            fp.write(self.model.to_json())
