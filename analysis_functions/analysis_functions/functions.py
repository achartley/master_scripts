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
