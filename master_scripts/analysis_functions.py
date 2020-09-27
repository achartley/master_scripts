import numpy as np
import json
import pandas as pd
from master_scripts.data_functions import (get_git_root, relative_energy,
                                           separation_distance,
                                           energy_difference,
                                           event_indices)
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops


def calc_kfold_accuracies(acc_list):
    """ Given a list of accuracies from a history object from model.fit,
    calculate mean accuracy, and get min and max accuracies for runs for
    one network.
    """

    # Find the top epoch acc for each fold
    best = []
    for fold in acc_list:
        val = np.asarray(fold)
        best.append(np.amax(val))
    best = np.asarray(best)

    # Calculate max, min, and mean accuracy
    acc_min = np.amin(best)
    acc_max = np.amax(best)
    acc_mean = np.mean(best)

    return [acc_min, acc_max, acc_mean]


def r2_score(y_true, y_pred):
    """ Given a set of true values and a set of predicted values,
    calculate the R2 score.
    """
    eps = 1e-13  # Epsilon avoid possible division by zero
    SS_res = np.sum(np.square(y_true - y_pred))
    SS_tot = np.sum(np.square(y_true - np.mean(y_true)))
    return (1 - SS_res / (SS_tot + eps))


def load_experiment(e_id):
    repo_root = get_git_root()
    e_path = repo_root + "experiments/"
    with open(e_path + e_id + ".json", "r") as fp:
        e = json.load(fp)
    return e


def load_hparam_search(name):
    """ Reads json-formatted hparam search file spec to pandas DF,
    and loads additional metrics into the dataframe.
    """
    hpath = get_git_root() + "experiments/searches/"
    df = pd.read_json(
        hpath + name, orient='index').rename_axis('id').reset_index()
    # JSON convert the tuples in hparam search to list when it's interpreted.
    # Convert the values to str to make it workable
    df['kernel_size'] = [str(x) for x in df['kernel_size'].values]
    # Add additional metrics to df
    accs = []
    f1 = []
    mcc = []
    auc = []
    for e_id in df['id']:
        e = load_experiment(e_id)
        accs.append(e['metrics']['accuracy_score'])
        f1.append(e['metrics']['f1_score'])
        mcc.append(e['metrics']['matthews_corrcoef'])
        auc.append(e['metrics']['roc_auc_score'])
    df['accuracy_score'] = accs
    df['f1_score'] = f1
    df['matthews_corrcoef'] = mcc
    df['roc_auc_score'] = auc
    return df


def double_event_indices(prediction, d_idx, c_idx):
    """Generates indices for correct and wrong classifications for double
    events, specifically. Indices for all doubles and events with small
    separation distances ('close doubles') are generated.

    param prediction: class predictions to generate indices for
    param d_idx: indices for all double events in the predictions
    param c_idx: indices for close double events in the predictions

    returns c_doubles: indices for all correct doubles
            w_doubles: indices for all wrong doubles
            c_close_doubles: indices for correct close doubles
            w_close_doubles: indices for wrong close doubles
    """
    c_doubles = np.where(prediction[d_idx] == 1)[0]
    w_doubles = np.where(prediction[d_idx] == 0)[0]
    c_close_doubles = np.where(prediction[c_idx] == 1)[0]
    w_close_doubles = np.where(prediction[c_idx] == 0)[0]

    return c_doubles, w_doubles, c_close_doubles, w_close_doubles


def singles_classification_stats(positions, energies, classification):
    """Outputs calculated separation distances, relative energies,
    and energy differences for double events in the dataset.


    :param positions:    event positions of interest, e.g validation positions
    :param energies:     event energies of interest, e.g validation energies
    :param classification:  event classification result

    :return df_singles: DataFrame containing information about single events
    """

    s_idx, d_idx, c_idx = event_indices(positions)
    df_singles = pd.DataFrame(
        data={
            "x_pos": positions[s_idx, 0].flatten(),
            "y_pos": positions[s_idx, 1].flatten(),
            "energy": energies[s_idx, 0].flatten(),
            "classification": classification[s_idx].flatten(),
            "indices": s_idx.flatten(),
        },
        index=np.arange(s_idx.shape[0])
    )
    return df_singles


def doubles_classification_stats(positions, energies, classification,
                                 close_max=1.0, scale=False):
    """Outputs calculated separation distances, relative energies,
    and energy differences for double events in the dataset.


    :param positions:    event positions of interest, e.g validation positions
    :param energies:     event energies of interest, e.g validation energies
    :param classification:  event classification result
    :param close_max:    Upper limit to what is to be considered a 'close'
                        event. Defaults to 1.0 pixels.

    :return df_doubles: DataFrame containing information about double events
    """

    s_idx, d_idx, c_idx = event_indices(positions, close_max)
    sep_dist = separation_distance(positions[d_idx])
    energy_diff = energy_difference(energies[d_idx])
    rel_energy = relative_energy(energies[d_idx], scale=scale)

    df_doubles = pd.DataFrame(
        data={
            "close": np.isin(d_idx, c_idx),
            "separation distance": sep_dist.flatten(),
            "relative energy": rel_energy.flatten(),
            "energy difference": energy_diff.flatten(),
            "classification": classification[d_idx].flatten(),
            "indices": d_idx.flatten(),
        },
        index=np.arange(d_idx.shape[0])
    )
    return df_doubles


def anodedata_classification_table(experiment_id, data_name,
                                   return_events=False):
    """Outputs the event dict post-classification as a table

    :param experiment_id:   unique id of experiment
    :param data_name:   filename of datafile without type suffix
                        ex. "anodedata_500k"
    :param return_events: bool, return the events dict if True
    """
    # Load the event classification results
    repo_root = get_git_root()
    fname = repo_root + "results/events_classified_" + data_name + "_"
    fname += experiment_id + ".json"
    with open(fname, "r") as fp:
        events = json.load(fp)

    # Generate list of unique event descriptors present in the events
    descriptors = list(
        set([event['event_descriptor'] for event in events.values()])
    )

    # Frequency of each type of descriptor for each event type
    desc_class = {
        'single': [],
        'double': [],
    }
    for event in events.values():
        desc_class[event['event_class']].append(event['event_descriptor'])

    # Translation dict for event descriptor
    # Note that not all of these correspond to something that may exists.
    translate_descriptor = {
        1: "Implant",
        2: "Decay",
        3: "implant + Decay",
        4: "Light ion",
        5: "Implant + Light Ion",
        6: "Decay + Light Ion",
        7: "Implant + Decay + Light Ion",
        8: "Double (time)",
        9: "Implant + Double (time)",
        10: "Decay + Double (time)",
        11: "Implant + Decay + Double (time)",
        12: "Light ion + Double (time)",
        13: "Implant + Light Ion + Double (time)",
        14: "Decay + Light ion + Double (time)",
        15: "Implant + Decay + Light Ion + Double (time)",
        16: "Double (space)",
        17: "Implant + Double (space)",
        18: "Decay + Double (space)"
    }

    # Print a table-like structure for viewing
    print("Classification results for {}:".format(experiment_id))
    print("|Event descriptor | Event type                   | singles | doubles |")
    print("| :---            |  :---:                       | :---:   | :---:   |")
    for d in descriptors:
        print("|{:^17d}|{:^30s}|{:^9d}|{:^9d}|".format(
            d,
            translate_descriptor[d],
            desc_class['single'].count(d),
            desc_class['double'].count(d)))

    if return_events:
        return events


def dsnt_mse(y_true, y_pred):
    """Computes the mean squared error between labels and predictions.
    After computing the squared distance between the inputs, the mean value
    over the last dimension is returned.
    `loss = mean(square(y_true - y_pred), axis=-1)`
    Standalone usage:
    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> assert np.array_equal(
    ...     loss.numpy(), np.mean(np.square(y_true - y_pred), axis=-1))
    Args:
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    Returns:
      Mean squared error values. shape = `[batch_size, d0, .. dN-1]`.
    """
    print("True, pre convert", y_true.shape)
    print("Pred, pre convert", y_pred.shape)
    y_pred = ops.convert_to_tensor_v2(y_pred[1])
    print("Pred, post convert", y_pred.shape)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    print("True, post convert", y_true.shape)
    exit(1)
    return K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1)
