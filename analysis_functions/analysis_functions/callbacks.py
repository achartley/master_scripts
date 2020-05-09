import tensorflow as tf
from tensorflow.keras import backend


class R2ScoreCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data

    def r2_score(y_true, y_pred):
        SS_res =  backend.sum(backend.square(y_true - y_pred))
        SS_tot = backend.sum(backend.square(y_true - backend.mean(y_true)))
        return ( 1 - SS_res/(SS_tot + backend.epsilon()) )


    def on_epoch_end(self, epoch, logs=None):
        y_pred = model.predict(model.x)
        r2 = r2_score(self.validation_data, y_pred)
        print("R2 score: ", r2)

