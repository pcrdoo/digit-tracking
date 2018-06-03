from __future__ import print_function
import h5py
from tensorflow.python import keras 
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras import backend as K
from digit_extraction import DigitExtractor
from skimage import img_as_float
from skimage.color import grey2rgb
import matplotlib.pyplot as plt
import cv2
import numpy as np

class MnistDataset:

    def aug_stretch(batch):
        pass

    def aug_pad(batch):
        pass

    def __init__(self, augmentation):
        # input image dimensions
        img_rows, img_cols = (28, 28)
        self.num_classes = 10

        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        # use floats
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # augment?
        if augmentation == 'stretch':
            x_train = aug_stretch(x_train)
            x_test = aug_stretch(x_test)
        elif augmentation == 'pad':
            x_train = aug_pad(x_train)
            x_test = aug_pad(x_test)
        else:
            # normalize
            x_train /= 255
            x_test /= 255
        
        # show first
        print(x_train[0])
        plt.imshow(x_train[0], cmap='binary')
        plt.show()

        # unsqueeze
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            self.shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            self.shape = (img_rows, img_cols, 1)

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        # save
        self.x_train, self.x_test = x_train, x_test 
        self.y_train, self.y_test = y_train, y_test