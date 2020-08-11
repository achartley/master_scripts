import numpy as np
import json
import pandas as pd
from master_scripts.data_functions import (get_git_root, relative_energy,
                                           separation_distance,
                                           energy_difference,
                                           event_indices)


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


def doubles_stats(indices, positions, energies):
    """Outputs calculated separation distances, relative energies,
    and energy differences as a pandas DataFrame.
    """

    sep_dist = separation_distance(positions[indices])
    energy_diff = energy_difference(energies[indices])
    rel_energy = relative_energy(energies[indices], scale=False)

    df = pd.DataFrame(
        data={
            "Separation distance": sep_dist.flatten(),
            "Relative energy": rel_energy.flatten(),
            "Energy difference": energy_diff.flatten(),
        },
        index=np.arange(indices.shape[0])
    )
    return df


def mean_values_doubles(indices, positions, energies, prediction):
    """Calculates mean values specific to double events and stores the results
    in a pandas dataframe.
    """
    s_idx, d_idx, c_idx = event_indices(positions[indices], threshold=1.0)
    sep_dist = separation_distance(positions[indices])
    energy_diff = energy_difference(energies[indices])
    rel_energy = relative_energy(energies[indices], scale=False)

    # Get indices
    (
        c_doubles,
        w_doubles,
        c_close_doubles,
        w_close_doubles
    ) = double_event_indices(prediction, d_idx, c_idx)

    # Mean distances
    mean_dist_all = np.mean(sep_dist[d_idx])
    mean_dist_c = np.mean(sep_dist[d_idx][c_doubles])
    mean_dist_w = np.mean(sep_dist[d_idx][w_doubles])
    mean_dist_close_c = np.mean(sep_dist[c_idx][c_close_doubles])
    mean_dist_close_w = np.mean(sep_dist[c_idx][w_close_doubles])

    # Mean relative energy
    mean_energy_all = np.mean(rel_energy[d_idx])
    mean_energy_c = np.mean(rel_energy[d_idx][c_doubles])
    mean_energy_w = np.mean(rel_energy[d_idx][w_doubles])
    mean_energy_close_c = np.mean(rel_energy[c_idx][c_close_doubles])
    mean_energy_close_w = np.mean(rel_energy[c_idx][w_close_doubles])

    # Mean energy difference
    mean_ediff_all = np.mean(energy_diff[d_idx])
    mean_ediff_c = np.mean(energy_diff[d_idx][c_doubles])
    mean_ediff_w = np.mean(energy_diff[d_idx][w_doubles])
    mean_ediff_close_c = np.mean(energy_diff[c_idx][c_close_doubles])
    mean_ediff_close_w = np.mean(energy_diff[c_idx][w_close_doubles])

    df_means = pd.DataFrame(
        data={
            "Separation distance [px]": [
                mean_dist_all,
                mean_dist_c,
                mean_dist_w,
                mean_dist_close_c,
                mean_dist_close_w
            ],
            "Relative energy": [
                mean_energy_all,
                mean_energy_c,
                mean_energy_w,
                mean_energy_close_c,
                mean_energy_close_w,
            ],
            "Energy difference": [
                mean_ediff_all,
                mean_ediff_c,
                mean_ediff_w,
                mean_ediff_close_c,
                mean_ediff_close_w,
            ]
        },
        index=[
            "All doubles",
            "Correct",
            "Wrong",
            "Correct close",
            "Wrong close",
        ]
    )

    return df_means
