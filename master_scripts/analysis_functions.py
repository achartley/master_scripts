import numpy as np
import json
import pandas as pd
from master_scripts.data_functions import get_git_root


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
    e_path = repo_root + "experiments/searches/"
    with open(e_path + e_id + ".json", "r") as fp:
        e = json.load(fp)
    return e


def load_hparam_search(name):
    """ Reads json-formatted hparam search file spec to pandas DF,
    and loads additional metrics into the dataframe.
    """
    repo_root = get_git_root()
    hpath = repo_root + "experiments/"
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
