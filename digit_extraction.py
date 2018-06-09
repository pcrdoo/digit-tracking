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
            return True, "N"

        q = w / h
        if q < MIN_SIDE_RATIO or q > MAX_SIDE_RATIO:
            return True, "D"

        if w * h > BIG_BOX_AREA_FRACTION * area:
            return True, "V"

        if sz < MIN_PIXELS:
            return True, "X"

        if w * h < MIN_AREA:
            return True, "P"

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
            return np.zeros([3,3])

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
            return np.zeros([3, 3])

        crop = roi[start[1]:end[1], start[0]:end[0]]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            return np.zeros([3, 3])

        thresh = threshold_sauvola(crop, window_size=13, k=0.025, r=0.5)
        return binary_opening(crop < thresh, selem=disk(1))

    def extract_digits(self, frame_o, w, h):
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
        cropped = [(resize(c, (h, w)), reason) if c is not None else
                None for c, reason in [(self.crop_rotrect(rect, frame_o), reason) for rect,reason in rects]]
        candidates = []
        cropped_with_indexes = [(c, i) for i, c in enumerate(cropped) if c is not None]
        for c, i in cropped_with_indexes:
            candidates.append(DigitCandidate(rects[i], c[0], c[1]))

        return candidates

    def draw_candidate(self, target, rect, image, confs, M = None, TL = None, reason = None):
        font = cv2.FONT_HERSHEY_SIMPLEX

        # center and box are used later for positioning
        rect = rect[0]
        center = rect[0]
        #print(rect)
        box = cv2.boxPoints(rect)

        # transform?
        if TL is not None:
            center = (center[0] + TL[0], center[1] + TL[1])
        if M is not None:
            center_r = np.asarray(center).reshape(1, 1, 2)
            center_t = cv2.perspectiveTransform(center_r, M)
            center = (center_t[0][0][0], center_t[0][0][1])
        if TL is not None:
            box[:, 0] += TL[0]
            box[:, 1] += TL[1]
        if M is not None:
            box_r = box.reshape(4, 1, 2)
            box_t = cv2.perspectiveTransform(box_r, M)
            box = box_t.reshape(4, 2)

        # we need ints
        center = np.int0(center)
        box = np.int0(box)

        # done, draw

        cv2.drawContours(target,[box],0,(0,0,1),2)

        x_off = int(center[0] - image.shape[1] / 2)
        y_off = int(center[1] - image.shape[0] / 2)
        try:
            target[y_off:y_off+image.shape[0], x_off:x_off+image.shape[1]] = grey2rgb(image)
        except ValueError:
            pass

        max_j = max(range(10), key=lambda j: confs[j])
        max_c = confs[max_j]

        cv2.putText(target,str(max_j),(int(center[0]) - 5, int(center[1]) - 20), font, 0.6,(1,0,0),2,cv2.LINE_AA)
        cv2.putText(target,reason,(int(center[0]) - 5, int(center[1]) + 20), font, 0.6,(1,0,0),2,cv2.LINE_AA)
