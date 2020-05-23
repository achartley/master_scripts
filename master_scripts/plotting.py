import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

# This plotting function is fetched from the scikit-learn's example
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_event(event, event_id):
    """ Plots a real even with all present info about the event.
    event_id: the id of the event.
    event: the dict associated with the event. Contains at least
            "event_descriptor" and "image".
            After classification and prediction additionally
            contains at least "event_class", "predicted_position",
            "predicted_energy".
    
    returns: ax object for the event.
    """
    fig, ax = plt.subplots()
    ax.imshow(event["image"].reshape((16,16)))
    ax.set_title(f"Event {event_id}")
    for i, key in enumerate(event.keys()):
        if key == "image":
            i -= 1
            continue
        ax.text(16, i, f"{key}: {event[key]}", fontsize=12)
        if key == "predicted_position":
            pos = event[key]
            ax.plot(pos[0], pos[1], 'rx')
    return ax

def plot_confusion_matrix(y_true, y_pred, classes,
        normalize=False,
        title=None,
        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # Set ticks and label them
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_title(title)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")


    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
            fig.tight_layout()
    return fig, ax

def plot_roc_curve(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    
    # Get the index of the threshold closest to 0.5
    idx = np.abs(thresholds - 0.5).argmin()

    # Plot the curve, with annotation for threshold ~0.5
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], 'k--')
    ax.plot(fpr, tpr)
    ax.annotate("threshold = 0.5", (fpr[idx], tpr[idx]),(0.2,0.6),
		arrowprops=dict(facecolor='black', shrink=0.05))
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('ROC curve')

    return fig, ax
