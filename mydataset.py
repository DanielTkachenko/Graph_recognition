import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import random as rd
import math_functions as mf

data_path = os.getcwd() + '\\data\\'
img_path = data_path + 'images\\'
labels_path = data_path + 'labels\\'

def get_rand_sequence(x, n):
    if n == 1:
        a = rd.uniform(-5., 5.)
        b = rd.uniform(-5., 5.)
        return [mf.linear(a, b, i) for i in x]
    elif n == 2:
        a = rd.uniform(-5., 5.)
        b = rd.uniform(-5., 5.)
        c = rd.uniform(-5., 5.)
        return [mf.quad(a, b, c, i) for i in x]
    elif n == 3:
        a = rd.uniform(-2., 2.)
        w = rd.uniform(-2., 2.)
        f = rd.uniform(-2., 2.)
        return [mf.sin_f(a, w, f, i) for i in x]
    elif n == 4:
        a = rd.uniform(-2., 2.)
        w = rd.uniform(-2., 2.)
        f = rd.uniform(-2., 2.)
        return [mf.cos_f(a, w, f, i) for i in x]
    elif n == 5:
        return [mf.root_n(i, 2) for i in x]
    else:
        return [mf.log_f(i) for i in x]


def generage_plot(n=2):
    for _ in range(n):
        x = np.arange(-10.0, 10.0, 0.1)
        i = rd.randint(1, 6)
        y = get_rand_sequence(x, i)
        plt.plot(x, y)
    plt.ylim([-20, 20])
    plt.axhline()
    plt.axvline()


def create_folders():
    if not os.path.exists(data_path):
        os.mkdir('data')
    else:
        print(' data folder already exists')

    if not os.path.exists(img_path):
        os.mkdir('data\\images')
    else:
        print(' images folder already exists')

    if not os.path.exists(labels_path):
        os.mkdir('data\\labels')
    else:
        print(' labels folder already exists')
    return 0

#execute it once to create dataset
def create_dataset(volume):
    create_folders()
    labels_file = open(labels_path + 'labels.txt', 'w')
    for i in range(volume):
        rand_n_plots = rd.randint(2, 5)
        generage_plot(rand_n_plots)
        img_name = str(i) + '.png'
        plt.savefig(img_path + img_name, dpi=50)
        labels_file.write(str(rand_n_plots) + '\n')
        plt.cla()
        plt.clf()
        plt.close()
    labels_file.close()

def load_data(train_percent=0.75):
    x = []
    files = os.listdir(path=img_path)
    for img_name in sorted(files):
        img = Image.open(img_path + img_name)
        x.append(np.array(img))
    with open(labels_path + 'labels.txt', 'r') as f:
        y = [int(i) for i in f.read().splitlines()]
    train_size = int(len(files) * train_percent)
    x_train, x_test = np.split(x, [train_size])
    y_train, y_test = np.split(y, [train_size])
    return x_train, x_test, y_train, y_test

#create_dataset(2000)