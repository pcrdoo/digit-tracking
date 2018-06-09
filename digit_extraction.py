import cv2
from skimage.filters import threshold_sauvola, threshold_otsu
from skimage import img_as_float
from skimage.morphology import binary_erosion, binary_closing, binary_opening, disk, binary_dilation
from skimage.measure import label, regionprops
from skimage.color import label2rgb, rgb2grey, grey2rgb
from skimage.transform import rescale, resize
from math import sqrt
import numpy as np

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

class DigitExtractor:
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
        size = rect[1]
        angle = rect[2]

        if size[1] < size[0]:
            angle += 90
            size = (size[1], size[0])

        pts = np.round(cv2.boxPoints(rect))
        #print(rect)
        #print(pts)
        bbox = cv2.boundingRect(pts)
        x, y, w, h = bbox
        if np.min(cv2.boxPoints(((x + w/2, y + h/2), (w, h), 0))) < 0:
            #print(rect)
            #print(bbox)
            #print('-')
            return None

        #print(img.shape)
        #print(bbox)
        #print('-')

        roi = img[y:y+h, x:x+w]

        new_center = (center[0] - x, center[1] - y)
        M = cv2.getRotationMatrix2D(new_center, angle, 1)
        roi = cv2.warpAffine(roi, M, (w, h))
        start = (int(new_center[0] - size[0] / 2)), int((new_center[1] - size[1] / 2))
        end = (int(new_center[0] + size[0] / 2)), int((new_center[1] + size[1] / 2))

        if start[1] - end[1] == 0 or start[0] - end[0] == 0:
            return None

        crop = roi[start[1]:end[1], start[0]:end[0]]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            return None

        return crop

    def extract_digits(self, frame_o):
        frame_o = img_as_float(frame_o)
        frame_color = frame_o
        frame_o = rgb2grey(frame_o)
        frame = rescale(frame_o, 0.5)
        gray = rgb2grey(frame)
        thresh = threshold_sauvola(gray, window_size=13, k=0.32, r=0.2)
        bin = gray > thresh
        dil = binary_closing(bin)
        dil = binary_erosion(bin, selem=disk(2))
        dil = 1.0 - dil
        cv2.imshow("dil",dil)
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

