import random

import cv2
import numpy as np
from skimage.measure import label
from tensorflow.python import keras
from tensorflow.python.keras import backend as K

from utils import transform_img


class MnistDataset:

    def pad(self, imgs):
        # List of padded images
        padded = []
        batch_sz = imgs.shape[0]
        for i in range(batch_sz):
            if i % 1000 == 0:
                print('Padding {}'.format(i))

            # Binarize
            img = imgs[i, :, :]
            bin = (img > 0).astype(int)
            wh_o = np.where(bin == 1)
            wh = (wh_o[1], wh_o[0])
            t = np.dstack(wh)[0]

            # Find bounding rect
            x, y, w, h = cv2.boundingRect(t)

            # Crop
            cropped = img[y:y+h, x:x+w]
            cropped2 = 1.0 - cropped

            # Transform
            trans = transform_img(cropped2, 28, 28, True)
            padded.append(trans)

        # Return the whole batch
        return np.stack(padded)

    # Very rough hack that has a significant impact on classification of ones
    def fix_one(self, img):
        # Binarize and find components
        ret, thresh = cv2.threshold(img, 1, 255, 0)
        lab, max_label = label(thresh, return_num=True)

        # Find number_of_pixels / minarearect_size ratio
        rat = 0
        largest_comp = 0
        for i in range(1, max_label+1):
            component_o = np.where(lab == i)
            component = (component_o[1], component_o[0])
            filled_px = component[0].size

            t = np.dstack(component)[0]
            rect = cv2.minAreaRect(t)
            box = np.int0(cv2.boxPoints(rect))
            ar = rect[1][0] * rect[1][1]

            if ar == 0:
                continue

            if filled_px > largest_comp:
                largest_comp = filled_px
                rat = filled_px / ar

        # If that ratio is large we can safely assume that this is a "one-line" one
        # Augment it to create a "two-line" one
        if rat > 0.9:
            x = np.min(box[:, 0])
            y = np.min(box[:, 1])
            center = (x, y)
            M = cv2.getRotationMatrix2D(center, -30, 1)
            img2 = cv2.warpAffine(
                img, M, img.shape, borderMode=cv2.BORDER_REPLICATE)
            mask = np.zeros(img.shape)
            mask[y:y+10, :] = 1
            img2 *= mask
            return np.maximum(img, img2)
        return None

    def __init__(self, augmentation):
        # Input image dimensions
        img_rows, img_cols = (28, 28)
        self.num_classes = 10

        # Split
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        # Use floats
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # Fix ones (only on train set, this will make final eval worse)
        nws = []
        all = []
        idxs = []
        for i in range(x_train.shape[0]):
            if y_train[i] == 1:
                idxs.append(i)
                all.append(x_train[i])
                img = x_train[i]
                nw = self.fix_one(img)
                if nw is not None:
                    nws.append(nw)

        # Sample and combine
        old_cnt = len(all) - len(nws)
        randIndex = random.sample(range(len(all)), old_cnt)
        randIndex.sort()
        sampled = [all[i] for i in randIndex]
        nws.extend(sampled)

        for i in range(len(idxs)):
            x_train[idxs[i]] = nws[i]

        # Normalize
        x_train /= 255
        x_test /= 255

        # Augment
        if augmentation == 'pad':
            x_train = self.pad(x_train)
            x_test = self.pad(x_test)

        # Unsqueeze
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            self.shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            self.shape = (img_rows, img_cols, 1)

        # Convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        # Save
        self.x_train, self.x_test = x_train, x_test
        self.y_train, self.y_test = y_train, y_test
