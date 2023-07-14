from keras.layers import Input, Dense, Reshape, Concatenate, Flatten, Lambda, Reshape, Conv2DTranspose
from keras.losses import binary_crossentropy
from keras.metrics import mse
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization
from keras.layers.activation import LeakyReLU
import numpy as np
import tensorflow as tf

import autoencoder

from tensorflow.python.framework.ops import disable_eager_execution

from image_tools import plot_digits, draw_manifold

disable_eager_execution()

class VAE(tf.keras.Model):
    def __init__(self, input_shape, latent_dim, batch_size, optimizer, filters=32, levels=2):
        super(VAE, self).__init__()
        self.img_shape = input_shape
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.optimizer = optimizer
        input_img = Input(shape=input_shape)
        x = autoencoder.build_encoder(input_img, levels, filters, (2, 2), True, False, False, True)
        x = Flatten()(x)
        self.z_mean = Dense(latent_dim)(x)
        self.z_log_var = Dense(latent_dim)(x)
        l = Lambda(self.sampling, output_shape=(latent_dim,))([self.z_mean, self.z_log_var])
        z = Input(shape=(latent_dim,))
        x = Dense(units=7*7*32, activation='relu')(z)
        x = Reshape(target_shape=(7, 7, 32))(x)
        x = autoencoder.build_decoder(x, levels, filters, (1, 1), False, False, False, True, (2, 2), True)
        x = Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same')(x)
        decoded = Reshape(target_shape=(input_shape))(x)
        self.encoder = Model(inputs=input_img, outputs=l)
        self.z_meaner = Model(inputs=input_img, outputs=self.z_mean)
        self.z_lvarer = Model(inputs=input_img, outputs=self.z_log_var)
        self.decoder = Model(inputs=z, outputs=decoded)
        self.vae = Model(inputs=input_img, outputs=self.decoder(self.encoder(input_img)))

    def sampling(self, args):
        z_mean, z_log_var = args
        print(z_mean)
        epsilon = K.random_normal(shape=(self.batch_size, self.latent_dim), mean=0., stddev=1.0)
        print(epsilon)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def vae_loss(self, x, decoded):
        x = K.reshape(x, shape=(self.batch_size, 28 * 28))
        decoded = K.reshape(decoded, shape=(self.batch_size, 28 * 28))
        xent_loss = 28 * 28 * binary_crossentropy(x, decoded)
        kl_loss = -0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        return (xent_loss + kl_loss) / 2 / 28 / 28

    def get_model(self):
        self.vae.compile(optimizer=self.optimizer, loss=self.vae_loss)
        return self.vae
