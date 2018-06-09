from tensorflow.python.keras.models import model_from_json
from tensorflow.python import keras
from tensorflow.python.keras import backend as K
from skimage.transform import rescale
from skimage.filters import threshold_sauvola
from skimage.morphology import binary_opening, disk, binary_dilation
from math import ceil, floor
import numpy as np
import json

class DigitClassifier():
    def __init__(self):
        self.num_classes = 10
        self.img_rows = 28
        self.img_cols = 28

        self.model = model_from_json(open('model/model.json', 'r').read())
        self.model.load_weights('model/model.h5')
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

    def transform_img(self, img):
        thresh = threshold_sauvola(img, window_size=13, k=0.025, r=0.5)
        img = img < thresh
        print(img.shape)

        [h, w] = img.shape
        if w > h:
            factor = self.img_cols / w
            print(factor)
            img = rescale(img, factor)
            print('by w')
            diff = (self.img_rows - img.shape[0]) / 2

            img = np.pad(img, ((int(ceil(diff)), int(floor(diff))), (0, 0)),
                    'constant', constant_values=((0,)))
        else:
            factor = self.img_rows / h
            print(factor)
            img = rescale(img, factor)
            print('by h',img.shape)
            diff = (self.img_cols - img.shape[1]) / 2

            img = np.pad(img, ((0, 0), (int(ceil(diff)), int(floor(diff)))),
                    'constant', constant_values=((0,)))

        return binary_dilation(img)

    def predict(self, imgs):
        # DO SOME IMG PROCESSING SO THE MODEL CAN ACCEPT IT

        # (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        # if K.image_data_format() == 'channels_first':
        # else:
        #     x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 1)
        # x_test = x_test.astype('float32')
        # x_test /= 255
        # y_test = keras.utils.to_categorical(y_test, self.num_classes)

        # img = x_test[:1]

        print(imgs.shape)
        imgs = imgs.reshape(imgs.shape[0], self.img_rows, self.img_cols, 1)
        return self.model.predict(imgs)
