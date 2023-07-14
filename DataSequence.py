import tensorflow as tf
import numpy as np
import math
import cv2

class CustomDataSequence(tf.keras.utils.Sequence):
    def __init__(self, x, y, batch_size):
        self.x, self.y = x, y
        self.batch_size = batch_size

    def __len__(self):
        math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, index):
        low = index * self.batch_size
        high = min(low + self.batch_size, len(self.x))
        batch_x = self.x[low:high]
        batch_y = self.y[low:high]
        return batch_x, batch_y

    def processing(self, file):
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.hight, self.weith))
        img = np.array(img).reshape(self.hight, self.weith, 3)
        img = img / 255.0
        img = self.augmentataion(img)
        return img

    def augmentation(self, image):
        image = tf.keras.layers.RandomFlip("horizontal")(image)
        image = tf.keras.layers.RandomRotation(0.1)(image)
        image = tf.keras.layers.RandomZoom(0.1)(image)
        return image