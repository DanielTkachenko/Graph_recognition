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

"""
#preparing data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.
x_test  = x_test .astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))
x_test  = np.reshape(x_test,  (len(x_test),  32, 32, 3))
#x_train_noised = get_noised_array(x_train, 0.3)
#x_test_noised = get_noised_array(x_test, 0.3)


#building neural network arcitecture
model = my_models.get_arch(input_shape=(32, 32, 3), levels=2, filters_min=8,
                           strides_shape_encoder=(2, 2),
                           batch_normalization=True,
                           conv2d_decoder=False,
                           max_pooling=False,
                           up_sampling=False,
                           conv2d_transpose=True,
                           c2dt_strides_shape=(2, 2))
#printing info about NN structure
#model.summary()

#start learning
model.fit(x_train, x_train,
                epochs=6,
                batch_size=254,
                shuffle=True,
                validation_data=(x_test, x_test),
          callbacks=[my_callbacks.CustomCallback()])


#demonstrating of saving and loading model on disc

#my_models.save_model(model, 'my_model1')
#loaded_model = my_models.load_model('my_model1')


#show results of autoencoder work
n = 10
imgs = x_test[:n]
#decoded_imgs = loaded_model.predict(imgs, batch_size=n)
decoded_imgs = model.predict(imgs, batch_size=n)
image_tools.plot_digits(imgs, decoded_imgs)
"""

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

"""
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
"""


#start learning
vae.fit(x_train, x_train, shuffle=True, epochs=10,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, x_test),
        callbacks=[my_callbacks.CustomVAECallback(vae=vae, generator=model.decoder,
                                                  x_test=x_test, latent_dim=LATENT_DIM, batch_size=BATCH_SIZE)],
        verbose=1)