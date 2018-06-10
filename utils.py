from skimage.transform import rescale
from skimage.filters import threshold_sauvola
from skimage.morphology import binary_dilation, binary_erosion, disk
from math import ceil, floor
import numpy as np
import cv2

# input: bgr=1 (white), digit=0 (black), grayscale
# output: bgr=0 (black), digi=1 (white), binary
def transform_img(img, img_rows, img_cols, mnist = False):
        # binarize
        thresh = threshold_sauvola(img, window_size=13, k=0.025, r=0.5)
        img = (img < thresh)

        [h, w] = img.shape

        # pad
        if w > h:
            factor = img_cols / w
            img = rescale(img, factor, mode='constant', cval=0)
            diff = (img_rows - img.shape[0]) / 2

            img = np.pad(img, ((int(ceil(diff)), int(floor(diff))), (0, 0)),
                    'constant', constant_values=((0,)))
        else:
            factor = img_rows / h
            img = rescale(img, factor, mode='constant', cval=0)
            diff = (img_cols - img.shape[1]) / 2

            img = np.pad(img, ((0, 0), (int(ceil(diff)), int(floor(diff)))),'constant', constant_values=((0,)))

        # binarize again
        if mnist:
            ret = (img > 0.4).astype(int) 
        else:
            ret = binary_dilation(img)
            ret = ret.astype(int)

        return ret