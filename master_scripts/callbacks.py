import tensorflow as tf
import numpy as np
from tensorflow.keras import backend


class R2ScoreCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super(R2ScoreCallback, self).__init__()
        self.val_x = validation_data[0]
        self.val_y = validation_data[1]

    """
    def r2_score(self, y_true, y_pred):
        SS_res =  backend.sum(backend.square(y_true - y_pred))
        SS_tot = backend.sum(backend.square(y_true - backend.mean(y_true)))
        return ( 1 - SS_res/(SS_tot + backend.epsilon()) )
    """
    def r2_score(self, y_true, y_pred):
        epsilon = np.finfo(np.float32).eps
        SS_res =  np.sum(np.square(y_true - y_pred))
        SS_tot = np.sum(np.square(y_true - np.mean(y_true)))
        return ( 1 - SS_res/(SS_tot + epsilon) )

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.val_x)
        r2 = self.r2_score(self.val_y, y_pred)
        print("val_r2: {:.4f}".format(r2))

