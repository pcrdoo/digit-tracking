from __future__ import print_function
import h5py
from tensorflow.python import keras 
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras import backend as K
from digit_extraction import DigitExtractor
from skimage import img_as_float
from skimage.color import label2rgb, rgb2grey, grey2rgb
import matplotlib.pyplot as plt
import cv2
import numpy as np
from utils import transform_img

class MnistDataset:

    def pad(self, imgs):
        # list of padded images
        padded = []
        batch_sz = imgs.shape[0]
        for i in range(batch_sz):
            if i % 1000 == 0:
                print(i)
            # binarize
            img = imgs[i,:,:]
            bin = (img > 0).astype(int)
            wh_o = np.where(bin == 1)
            wh = (wh_o[1], wh_o[0])
            sz = wh[0].size 
            t = np.dstack(wh)[0]

            # find bounding rect
            x,y,w,h = cv2.boundingRect(t)

            # crop
            cropped = img[y:y+h, x:x+w]
            cropped2 = 1.0 - cropped

            # transform
            trans = transform_img(cropped2, 28, 28, True)
            padded.append(trans)
        return np.stack(padded)

    def __init__(self, augmentation):
        # input image dimensions
        img_rows, img_cols = (28, 28)
        self.num_classes = 10

        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        # use floats
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # for i in range()



        img = x_train[3]
        print(img.shape)
        img_rgb = grey2rgb(img).astype(np.uint8)
        print(img_rgb.shape)
        edges = cv2.Canny(img_rgb, 50, 150)
        #plt.imshow(edges, cmap='gist_gray')
        #plt.show()
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, minLineLength=90, maxLineGap=20)
        print(lines)

        # HEY


        # normalize
        x_train /= 255
        x_test /= 255

        arr = dict()
        for i in range(60000):
            if y_train[i] == 1:
                print(i)
                plt.imshow(x_train[i], cmap='gist_gray')
                plt.colorbar()
                plt.show()
                zzz = input()
            if y_train[i] not in arr:
                arr[y_train[i]] = 0
            arr[y_train[i]] += 1
        print(arr)

        # augment?
        if augmentation == 'pad':
            x_train = self.pad(x_train)
            x_test = self.pad(x_test)

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
    
mn = MnistDataset('pad')