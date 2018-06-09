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
    def __init__(self, img_rows, img_cols):
        self.num_classes = 10
        self.img_rows = img_rows 
        self.img_cols = img_cols 

        self.model = model_from_json(open('model/model.json', 'r').read())
        self.model.load_weights('model/model.h5')
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

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

        # print(imgs.shape)
        imgs = imgs.reshape(imgs.shape[0], self.img_rows, self.img_cols, 1)
        return self.model.predict(imgs)
