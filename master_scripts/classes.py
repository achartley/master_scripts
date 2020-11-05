from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (matthews_corrcoef, f1_score, confusion_matrix,
                             roc_auc_score, accuracy_score, r2_score,
                             mean_squared_error, mean_absolute_error)
from master_scripts.data_functions import get_git_root, normalize_image_data
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
            x=normalize_image_data(x[train_idx]),
            y=y[train_idx],
            validation_data=(
                normalize_image_data(x[val_idx]),
                y[val_idx]
            ),
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
        It is assumed that the input data is preprocessed, normalized,
        and good to go.

        :param x:   input data (batch_size, width, height, channels)
        :param y:   labels / targets
        """

        # Store accuracy for each fold for all models
        results = {}

        # Create KFold data generator
        kf = KFold(
            random_state=self.config['random_seed'],
            **self.config['kfold_args']
        )

        # Run k-fold cross-validation
        fold = 0  # Track which fold
        for train_idx, val_idx in kf.split(x, y):

            # Train model
            history = self.model.fit(
                x=normalize_image_data(x[train_idx]),
                y=y[train_idx],
                validation_data=(
                    normalize_image_data(x[val_idx]),
                    y[val_idx]
                ),
                ** self.config['fit_args'],
            ).history
            # Calculate metrics for the model
            if self.model_type == "classification":
                self.classification_metrics(x[val_idx], y[val_idx], fold)
            elif self.model_type == "regression":
                self.regression_metrics(x[val_idx], y[val_idx], fold)

            # Store train and val indices for the fold
            foldkey = 'fold_' + str(fold)
            print("Now in fold: ", foldkey)
            self.indices[foldkey] = {
                'train_idx': train_idx.tolist(),
                'val_idx': val_idx.tolist(),
            }
            # Store the history object
            results[foldkey] = history

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
        if fold is not None:
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
        y_out = self.model.predict(normalize_image_data(x_val))
        y_pred = y_out > 0.5
        confmat = confusion_matrix(y_val, y_pred)

        metrics = {}
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

        if fold is not None:
            foldkey = 'fold_' + str(fold)
            self.metrics_kfold[foldkey] = metrics
        else:
            self.metrics = metrics

    def save(self, save_model=False, save_indices=False):
        """ Outputs two files:
        - one experiment config with performance metrics and optimizer
        information, fit parameters etc
        - model config, loadable with tf.keras.models.model_from_json.

        :param save_model: Boolean. Save complete model with weights if true.
        :param save_indices: Boolean. Save indices to file if True.
        """

        # Collect information into a dictionary
        output = {}
        output['loss'] = self.model.loss
        output['optimizer'] = str(self.model.optimizer.get_config())
        output['experiment_config'] = self.config
        output['model_type'] = self.model_type
        output['experiment_name'] = self.experiment_name
        output['experiment_id'] = self.id
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

        if save_indices:
            output['indices'] = self.indices
        # Save the model with weights
        if save_model:
            mpath = self.config['path_args']['models'] + self.id + ".h5"
            self.model.save(mpath)

        # Save experiment
        with open(experiment_fpath, 'w') as fp:
            if save_indices:
                json.dump(output, fp)
            else:
                json.dump(output, fp, indent=2)

        # Save model config
        with open(model_fpath, 'w') as fp:
            fp.write(self.model.to_json())


class DSNT(tf.keras.layers.Layer):
    """
    Differentiable Spatial to Numerical Transform, as taken from the paper
    "Numerical Coordinate Regression with Convolutional Neural Networks".
    Implemented as a subclass of a Keras Layer.
    Paper: https://arxiv.org/pdf/1801.07372.pdf
    Original implementation: https://github.com/ashwhall/dsnt
    """

    def __init__(self, name=None, **kwargs):
        super(DSNT, self).__init__(name=name, **kwargs)

    def call(self, inputs, method="softmax"):
        '''
        Differentiable Spatial to Numerical Transform, as taken from the paper
        "Numerical Coordinate Regression with Convolutional Neural Networks"
        Arguments:
            inputs - The learnt heatmap. A 3d tensor of shape
                    [batch, height, width]
            method - A string representing the normalisation method.
                    See `normalise_heatmap` for available methods
        Returns:
            norm_heatmap - The given heatmap with normalisation/rectification
                        applied
            coords_zipped - A tensor of shape [batch, 2] containing the [x, y]
                        coordinate pairs
        '''

        def js_reg_loss(heatmaps, centres, fwhm=1):
            '''
            Calculates and returns the average Jensen-Shannon divergence
            between heatmaps and target Gaussians.
            Arguments:
                heatmaps - Heatmaps generated by the model
                entres - Centres of the target Gaussians (in normalized units)
                fwhm - Full-width-half-maximum for the drawn Gaussians, which
                can be thought of as a radius.
            '''
            gauss = make_gaussians(centres, tf.shape(
                heatmaps)[1], tf.shape(heatmaps)[2], fwhm)
            divergences = _js_2d(heatmaps, gauss)
            return tf.reduce_mean(divergences)

        def normalise_heatmap(inputs, method='softmax'):
            '''
            Applies the chosen normalisation/rectification method to the input
            tensor
            Arguments:
                inputs - A 4d tensor of shape [batch, height, width, 1]
                method - A string representing the normalisation method.
                        One of those shown below
            '''
            # Remove the final dimension as it's of size 1
            inputs = tf.reshape(inputs, tf.shape(inputs)[:3])

            # Normalise values so the values sum to one for each heatmap
            def normalise(x): return tf.div(
                x, tf.reshape(tf.reduce_sum(x, [1, 2]), [-1, 1, 1]))

            # Perform rectification
            if method == 'softmax':
                inputs = _softmax2d(inputs, axes=[1, 2])
            elif method == 'abs':
                inputs = tf.abs(inputs)
                inputs = normalise(inputs)
            elif method == 'relu':
                inputs = tf.nn.relu(inputs)
                inputs = normalise(inputs)
            elif method == 'sigmoid':
                inputs = tf.nn.sigmoid(inputs)
                inputs = normalise(inputs)
            else:
                msg = "Unknown rectification method \"{}\"".format(method)
                raise ValueError(msg)
            return inputs

        def _kl_2d(p, q, eps=24):
            unsummed_kl = p * (tf.log(p + eps) - tf.log(q + eps))
            kl_values = tf.reduce_sum(unsummed_kl, [-1, -2])
            return kl_values

        def _js_2d(p, q, eps=1e-24):
            m = 0.5 * (p + q)
            return 0.5 * _kl_2d(p, m, eps) + 0.5 * _kl_2d(q, m, eps)

        def _softmax2d(target, axes):
            '''
            A softmax implementation which can operate across more than one
            axis as this isn't provided by Tensorflow
            Arguments:
                targets - The tensor on which to apply softmax
                axes - An integer or list of integers across which to apply
                        softmax
            '''
            max_axis = tf.reduce_max(target, axes, keepdims=True)
            target_exp = tf.exp(target - max_axis)
            normalize = tf.reduce_sum(target_exp, axes, keepdims=True)
            softmax = target_exp / normalize
            return softmax

        def make_gaussian(size, centre, fwhm=1):
            '''
            Makes a rectangular gaussian kernel.
            Arguments:
                size    - A 2d tensor representing [height, width]
                centre  - Pair of (normalised [0, 1]) x, y coordinates
                fwhm    - Full-width-half-maximum, which can be thought of as a
                            radius.
            '''
            # Scale normalised coords to be relative to the size of the frame
            centre = [centre[0] * tf.cast(size[1], tf.float32),
                      centre[1] * tf.cast(size[0], tf.float32)]
            # Find largest side as we build a square and crop to desired size
            square_size = tf.cast(tf.reduce_max(size), tf.float32)

            x = tf.range(0, square_size, 1, dtype=tf.float32)
            y = x[:, tf.newaxis]
            x0 = centre[0] - 0.5
            y0 = centre[1] - 0.5
            unnorm = tf.exp(
                -4 * tf.log(2.) * ((x - x0)**2 + (y - y0) ** 2) / fwhm**2
            )[:size[0], :size[1]]
            norm = unnorm / tf.reduce_sum(unnorm)
            return norm

        def make_gaussians(centres_in, height, width, fwhm=1):
            '''
            Makes a batch of gaussians. Size of images designated by
            height, width; number of images designated by length of the 1st
            dimension of centres_in
            Arguments:
                centres_in  - The normalised coordinate centres of the
                                gaussians of shape [batch, x, y]
                height  - The desired height of the produced gaussian image
                width   - The desired width of the produced gaussian image
                fwhm    - Full-width-half-maximum, which can be thought of as
                                a radius.
            '''
            def cond(centres, heatmaps):
                return tf.greater(tf.shape(centres)[0], 0)

            def body(centres, heatmaps):
                curr = centres[0]
                centres = centres[1:]
                new_heatmap = make_gaussian([height, width], curr, fwhm)
                new_heatmap = tf.reshape(new_heatmap, [-1])

                heatmaps = tf.concat([heatmaps, new_heatmap], 0)
                return [centres, heatmaps]

            # Produce 1 heatmap per coordinate pair, build a list of heatmaps
            _, heatmaps_out = tf.while_loop(
                cond,
                body,
                [centres_in, tf.constant([])],
                shape_invariants=[
                    tf.TensorShape([None, 2]),
                    tf.TensorShape([None])]
            )
            heatmaps_out = tf.reshape(heatmaps_out, [-1, height, width])
            return heatmaps_out

        def get_config(self):
            config = super(DSNT, self).get_config()
            return config

        # Rectify and reshape inputs
        norm_heatmap = normalise_heatmap(inputs, method)

        batch_count = tf.shape(norm_heatmap)[0]
        height = tf.shape(norm_heatmap)[1]
        width = tf.shape(norm_heatmap)[2]

        # Build the DSNT x, y matrices
        dsnt_x = tf.tile(
            [[(2 * tf.range(1, width + 1) - (width + 1)) / width]],
            [batch_count, height, 1]
        )
        dsnt_x = tf.cast(dsnt_x, tf.float32)
        dsnt_y = tf.tile(
            [[(2 * tf.range(1, height + 1) - (height + 1)) / height]],
            [batch_count, width, 1]
        )
        dsnt_y = tf.cast(tf.transpose(dsnt_y, perm=[0, 2, 1]), tf.float32)

        # Compute the Frobenius inner product
        outputs_x = tf.reduce_sum(tf.multiply(
            norm_heatmap, dsnt_x), axis=[1, 2])
        outputs_y = tf.reduce_sum(tf.multiply(
            norm_heatmap, dsnt_y), axis=[1, 2])

        # Zip into [x, y] pairs
        coords_zipped = tf.stack([outputs_x, outputs_y], axis=1)

        return norm_heatmap, coords_zipped
