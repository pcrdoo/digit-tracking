from skimage.transform import rescale
from skimage.filters import threshold_sauvola
from skimage.morphology import binary_dilation
from math import ceil, floor
import numpy as np

def transform_img(img, img_rows, img_cols):
        thresh = threshold_sauvola(img, window_size=13, k=0.025, r=0.5)
        img = img < thresh
        # print(img.shape)

        [h, w] = img.shape
        if w > h:
            factor = img_cols / w
            #print(factor)
            img = rescale(img, factor)
            #print('by w')
            diff = (img_rows - img.shape[0]) / 2

            img = np.pad(img, ((int(ceil(diff)), int(floor(diff))), (0, 0)),
                    'constant', constant_values=((0,)))
        else:
            factor = img_rows / h
            #print(factor)
            img = rescale(img, factor)
            #print('by h',img.shape)
            diff = (img_cols - img.shape[1]) / 2

            img = np.pad(img, ((0, 0), (int(ceil(diff)), int(floor(diff)))),
                    'constant', constant_values=((0,)))

        return binary_dilation(img)