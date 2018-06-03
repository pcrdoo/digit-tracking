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
    def __init__(self, rect, image):
        self.rect = rect
        self.image = image

class DigitExtractor:
    def rect_eligible(self, rect, sz, area):
        w, h = rect[1]
        if w > h:
            w, h = h, w

        if w == 0 or h == 0:
            return False

        q = w / h
        if q < MIN_SIDE_RATIO or q > MAX_SIDE_RATIO:
            return False

        if w * h > BIG_BOX_AREA_FRACTION * area:
            return False

        if sz < MIN_PIXELS:
            return False

        if w * h < MIN_AREA:
            return False

        return True

    def crop_rotrect(self, rect, img):
        center = rect[0]
        size = rect[1]
        angle = rect[2]

        if size[1] < size[0]:
            angle += 90
            size = (size[1], size[0])

        rectpoints = cv2.boxPoints(rect)
        M = cv2.getRotationMatrix2D(center, -angle, 1)
        bb = cv2.boundingRect(rectpoints)
        bbpoints = cv2.boxPoints(((bb[0], bb[1]), (bb[2], bb[3]), 0.0))
        rectpoints_rot = np.int0(cv2.transform(np.array([rectpoints]), M))[0]
        govno=np.concatenate((bbpoints, rectpoints_rot)).astype(np.float32)
        bb = cv2.boundingRect(govno)

        x, y, w, h = bb

        roi = img[y:y+h, x:x+w]
        rows, cols = roi.shape[0], roi.shape[1]
        if rows <= 0 or cols <= 0:
            return None

        origin = (center[0] - x, center[1] - y)
        M = cv2.getRotationMatrix2D(origin, angle, 1)
        rot = cv2.warpAffine(roi, M, (cols, rows))

        rect0 = (origin, size, angle)
        box = cv2.boxPoints(rect0)
        pts = np.int0(cv2.transform(np.array([box]), M))[0]
        pts[pts < 0] = 0

        # crop
        crop = rot[pts[1][1]:pts[0][1],
                   pts[1][0]:pts[2][0]]

        thresh = threshold_sauvola(crop, window_size=13, k=0.025, r=0.5)
        return binary_opening(crop < thresh, selem=disk(1))

    def extract_digits(self, frame_o, w, h):
        frame_o = img_as_float(frame_o)
        frame_color = frame_o
        frame_o = rgb2grey(frame_o)
        frame = rescale(frame_o, 0.5)
        gray = rgb2grey(frame)
        thresh = threshold_sauvola(gray, window_size=17, k=0.04, r=0.2)
        bin = gray > thresh
        dil = binary_closing(bin)
        dil = binary_erosion(bin, selem=disk(2))
        dil = 1.0 - dil
        lab, num = label(dil, return_num=True)

        if not num:
            return

        components = []
        for i in range(num):
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
            if self.rect_eligible(rect, sz, area):
                rects.append(rect)

        cropped = [resize(c, (h, w)) if c is not None else
                None for c in [self.crop_rotrect(rect, frame_o) for rect in rects]]
        candidates = []
        cropped_with_indexes = [(c, i) for i, c in enumerate(cropped) if c is not None]
        for c, i in cropped_with_indexes:
            candidates.append(DigitCandidate(rects[i], c))

        return candidates

    def draw_candidate(self, target, rect, image, confs, M = None, TL = None):
        font = cv2.FONT_HERSHEY_SIMPLEX

        # center and box are used later for positioning
        center = rect[0]
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
