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

    from skimage.measure import label, regionprops
    def fix_one(self, img):

        ret,thresh = cv2.threshold(img,1,255,0)
        # HEY 
        lab, max_label = label(thresh, return_num=True)

        rat = 0
        largest_comp = 0
       
        for i in range(1, max_label+1):
            component_o = np.where(lab == i)
            component = (component_o[1], component_o[0])
            filled_px = component[0].size

            sz = component[0].size
            t = np.dstack(component)[0]
            rect = cv2.minAreaRect(t)
            box = np.int0(cv2.boxPoints(rect))
            ar = rect[1][0] * rect[1][1]

            if ar == 0:
                continue

            if filled_px > largest_comp:
                largest_comp = filled_px
                rat = filled_px / ar

        if rat > 0.9:
            # surely we can augment
            # augment now
            x = np.min(box[:, 0])
            y = np.min(box[:, 1])
            center = (x, y)
            M = cv2.getRotationMatrix2D(center, -30, 1)
            img2 = cv2.warpAffine(img, M, img.shape, borderMode=cv2.BORDER_REPLICATE)

            mask = np.zeros(img.shape)
            mask[y:y+10, :] = 1
            img2 *= mask


            img3 = np.maximum(img, img2)

            return img3

        return None

    def __init__(self, augmentation):
        # input image dimensions
        img_rows, img_cols = (28, 28)
        self.num_classes = 10

        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        # use floats
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # FIX ONES
        nws = []
        all = []
        idxs = []
        for i in range(60000):
            if y_train[i] == 1:
                idxs.append(i)
                all.append(x_train[i])
                img = x_train[i]
                nw = self.fix_one(img) #new_stuff
                if nw is not None:
                    nws.append(nw)
        # sample!
        old_cnt = len(all) - len(nws)
        import random
        randIndex = random.sample(range(len(all)), old_cnt)
        randIndex.sort()
        sampled = [all[i] for i in randIndex]
        nws.extend(sampled)

        for i in range(len(idxs)):
            x_train[idxs[i]] = nws[i]

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