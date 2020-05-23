import numpy as np


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
    eps = 1e-13 # Epsilon avoid possible division by zero
    SS_res =  np.sum(np.square(y_true - y_pred))
    SS_tot = np.sum(np.square(y_true - np.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + eps) )
