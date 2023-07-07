import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import cifar10
import tensorflow as tf
import my_models
import my_callbacks

def plot_digits(*args):
    args = [x.squeeze() for x in args]
    n = min([x.shape[0] for x in args])

    plt.figure(figsize=(2 * n, 2 * len(args)))
    for j in range(n):
        for i in range(len(args)):
            ax = plt.subplot(len(args), n, i * n + j + 1)
            plt.imshow(args[i][j])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()

def addsalt_pepper(img, SNR):
    img_ = img.copy()
    c, h, w = img_.shape
    mask = np.random.choice((0, 1, 2), size=(1, h, w), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
    mask = np.repeat (mask, c, axis = 0) # Копировать по каналу в ту же форму, что и img
    img_ [mask == 1] = 255 # солевой шум
    img_ [mask == 2] = 0 # перцовый шум
    return img

def get_noised_array(x, snr):
    x_noised = x.copy()
    for img in x_noised:
        img = addsalt_pepper(img, snr)
    return x_noised

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
"""
my_models.save_model(model, 'my_model1')
loaded_model = my_models.load_model('my_model1')
"""

#show results of autoencoder work
n = 10
imgs = x_test[:n]
#decoded_imgs = loaded_model.predict(imgs, batch_size=n)
decoded_imgs = model.predict(imgs, batch_size=n)
plot_digits(imgs, decoded_imgs)
