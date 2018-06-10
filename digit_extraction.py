import cv2
from skimage.filters import threshold_sauvola, threshold_otsu
from skimage import img_as_float
from skimage.morphology import binary_erosion, binary_closing, binary_opening, disk, binary_dilation
from skimage.measure import label, regionprops
from skimage.color import label2rgb, rgb2grey, grey2rgb
from skimage.transform import rescale, resize
from skimage.exposure import equalize_hist
from math import sqrt
import numpy as np
import math 

# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
MIN_SIDE_RATIO = 0.20
MAX_SIDE_RATIO = 1.30
MIN_AREA = 14 * 14
MIN_PIXELS = 50
BIG_BOX_AREA_FRACTION = 0.3

class DigitCandidate:
    def __init__(self, rect, image, reason=None):
        self.rect = rect
        self.image = image
        self.reason = reason

        self.guess = -1 
        self.conf = 1
        self.shaprness = 0

class DigitExtractor:
    def __init__(self):
        self._k = 0.45
        self._r = 1.02
        self._ws = 11

    def rect_eligible(self, rect, sz, area):
        w, h = rect[1]
        if w > h:
            w, h = h, w

        if w == 0 or h == 0:
            return False, "N"
        
        if w < 4 or h < 4:
            return False, "F"

        q = w / h
        if q < MIN_SIDE_RATIO or q > MAX_SIDE_RATIO:
            return False, "D"

        if w * h > BIG_BOX_AREA_FRACTION * area:
            return False, "V"

        if sz < MIN_PIXELS:
            return False, "X"

        if w * h < MIN_AREA:
            return False, "P"

        return True, "O"
    
    def crop_rotrect(self, rect, img):
        center = rect[0]
        rect_x, rect_y = center
        size = rect[1]
        rect_w, rect_h = size
        angle = rect[2]

        if angle < -45:
            ang_rot = 90 + angle 
        else:
            ang_rot = angle

        pts = np.round(cv2.boxPoints(rect))
        bbox = cv2.boundingRect(pts)
        x, y, w, h = bbox
        
        if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
            return None

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x + w > img.shape[1]:
            w = img.shape[1] - x
        if y + h > img.shape[0]:
            h = img.shape[0] - y

        roi = img[y:y+h, x:x+w]

        new_center = (rect_x - x, rect_y - y)
        new_c_x, new_c_y = new_center

        M = cv2.getRotationMatrix2D(new_center, ang_rot, 1)

        #cv2.imshow('roi pre', roi)

        roi = cv2.warpAffine(roi, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        #cv2.imshow('roi post', roi)

        # rotate bounding box
        box = cv2.boxPoints((new_center, size, angle))
        pts = np.int0(cv2.transform(np.array([box]), M))[0]    
        pts[pts < 0] = 0

        first_row = np.min(pts[:, 1])
        last_row = np.max(pts[:, 1])
        first_col = np.min(pts[:, 0])
        last_col = np.max(pts[:, 0])

        img_crop = roi[first_row:last_row,first_col:last_col]
        
        return img_crop

    def extract_digits(self, frame_o):
        frame_o = img_as_float(frame_o)
        frame_o = rgb2grey(frame_o)
        frame = rescale(frame_o, 0.5)
        gray = rgb2grey(frame)
        cv2.imshow('gray', gray)

        gray = equalize_hist(gray)
        cv2.imshow('gray2', gray)


        thresh = threshold_sauvola(gray, window_size=self._ws, k=self._k, r=self._r)
        bin = gray > thresh
        dil = binary_closing(bin)
        dil = binary_erosion(bin, selem=disk(2))
        dil = 1.0 - dil
        cv2.imshow('dil', dil)
        lab, max_label = label(dil, return_num=True)

        if not max_label:
            return

        components = []
        for i in range(1, max_label + 1):
            component_o = np.where(lab == i)
            component = (component_o[1], component_o[0])
            sz = component[0].size
            components.append(component)

        rects = []
        area = dil.shape[0] * dil.shape[1]
        for c in components:
            sz = c[0].size
            t = np.dstack(c)[0] * 2
            rect = cv2.minAreaRect(t)
            ret,reason= self.rect_eligible(rect, sz, area)
            if ret:
                #print(rect)
                rects.append((rect, reason))


        #print('*****')
        cropped = [(c, reason) if c is not None else
                None for c, reason in [(self.crop_rotrect(rect, frame_o), reason) for rect,reason in rects]]
        candidates = []
        cropped_with_indexes = [(c, i) for i, c in enumerate(cropped) if c is not None]
        for c, i in cropped_with_indexes:
            candidates.append(DigitCandidate(rects[i], c[0], c[1]))

        return candidates

