import numpy as np
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from keras.datasets import cifar10
import tensorflow as tf
from IPython.display import clear_output
from keras.callbacks import LambdaCallback, ReduceLROnPlateau, TensorBoard

import autoencoder
import my_callbacks
import image_tools
from VAE import VAE

# гиперпараметры
VARIATIONAL = True
HEIGHT = 28
WIDTH = 28
BATCH_SIZE = 500
LATENT_DIM = 2
DROPOUT_RATE = 0.3
START_FILTERS = 32
CAPACITY = 3
CONDITIONING = True
OPTIMIZER = Adam(lr=0.01)

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)) / 255.
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)) / 255.

model = VAE(input_shape=(28, 28, 1), latent_dim=LATENT_DIM, batch_size=BATCH_SIZE, optimizer=OPTIMIZER)
vae = model.get_model()


#integration with wandb
import wandb
wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project="Sputnik-Internship",
    # Track hyperparameters and run metadata
    config={
        "loss": model.vae_loss,
        "metric": "accuracy",
        "epoch": 10,
        "batch_size": BATCH_SIZE
    })



#start learning
vae.fit(x_train, x_train, shuffle=True, epochs=10,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, x_test),
        callbacks=[my_callbacks.CustomVAECallback(vae=vae, generator=model.decoder,
                                                  x_test=x_test, latent_dim=LATENT_DIM, batch_size=BATCH_SIZE)],
        verbose=1)
