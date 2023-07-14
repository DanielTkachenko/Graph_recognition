import keras.callbacks
from matplotlib import pyplot as plt
from IPython.display import clear_output
import numpy as np

import image_tools


class CustomAECallback(keras.callbacks.LambdaCallback):
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

class CustomVAECallback(keras.callbacks.LambdaCallback):
    def __init__(self, vae, generator, x_test, latent_dim, batch_size):
        self.vae = vae
        self.generator = generator
        self.x_test = x_test
        self.latent_dim = latent_dim
        self.batch_size = batch_size

    def on_epoch_begin(self, epoch, logs=None):
        return
    def on_epoch_end(self, epoch, logs=None):
        clear_output()
        imgs = self.x_test[:self.batch_size]
        # wandb logging
        # wandb.log({"epoch": logs["epoch"], "loss": logs["loss"]})
        # prediction
        decoded = self.vae.predict(imgs, batch_size=self.batch_size)
        image_tools.plot_digits(imgs[:10], decoded[:10])
        # Рисование многообразия
        figure = image_tools.draw_manifold(self.generator, digit_size=28, n=10, latent_dim=self.latent_dim, show=True)
    def on_batch_begin(self, batch, logs=None):
        return
    def on_batch_end(self, batch, logs=None):
        return
    def on_train_begin(self, logs=None):
        return
    def on_train_end(self, logs=None):
        return