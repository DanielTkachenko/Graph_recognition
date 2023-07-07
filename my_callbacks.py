import keras.callbacks
from matplotlib import pyplot as plt


class CustomCallback(keras.callbacks.LambdaCallback):
    def __init__(self):
        self.batch_list = []
        self.loss_list = []
        self.alpha = 0

    def on_epoch_begin(self, epoch, logs=None):
        return
    def on_epoch_end(self, epoch, logs=None):
        self.alpha = len(self.batch_list)
    def on_batch_begin(self, batch, logs=None):
        return
    def on_batch_end(self, batch, logs=None):
        self.batch_list.append(batch + self.alpha)
        self.loss_list.append(logs['loss'])
    def on_train_begin(self, logs=None):
        return
    def on_train_end(self, logs=None):
        plt.plot(self.batch_list, self.loss_list)
        plt.show()
