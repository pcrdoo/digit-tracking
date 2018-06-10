import cv2
from skimage.filters import threshold_sauvola, threshold_otsu
from skimage import img_as_float, img_as_ubyte
from skimage.morphology import binary_erosion, binary_closing, binary_opening, disk, binary_dilation
from skimage.measure import label, regionprops, compare_ssim
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
MAGICNA_PATKA = 200

class DigitCandidate:
    def __init__(self, rect, image, reason=None):
        self.rect = rect
        self.image = image
        self.reason = reason

        self.guess = -1 
        self.conf = 1
        self.sharpness = 0
        self.dimensions = rect[1]

    def __repr__(self):
        return "{R = " + repr(self.rect) + "}"

class DigitExtractor:
    def __init__(self, IMSHOW_DBG):
        self.IMSHOW_DBG = IMSHOW_DBG
        self._k = 0.45
        self._r = 1.02
        self._ws = 11
        self.tracked = False

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

    def rotrects_from_image(self, bin, scale=1, filter=True):
        lab, max_label = label(bin, return_num=True)

        if not max_label:
            return []

        components = []
        for i in range(1, max_label + 1):
            component_o = np.where(lab == i)
            component = (component_o[1], component_o[0])
            sz = component[0].size
            components.append(component)

        rects = []
        area = bin.shape[0] * bin.shape[1]
        for c in components:
            sz = c[0].size
            t = np.dstack(c)[0] * scale
            rect = cv2.minAreaRect(t)
            if not filter or self.rect_eligible(rect, sz, area):
                rects.append(rect)

        return rects

    def preprocess_image(self, gray):
        gray = equalize_hist(gray)

        if self.IMSHOW_DBG:
            cv2.imshow('gray2', gray)

        thresh = threshold_sauvola(gray, window_size=self._ws, k=self._k, r=self._r)
        bin = gray > thresh
        dil = binary_closing(bin)
        dil = binary_erosion(bin, selem=disk(2))
        dil = 1.0 - dil

        return dil

    def preprocess_roi(self, roi):
        thresh = threshold_sauvola(roi, window_size=5, k=0.13)
        bin = roi > thresh
        dil = binary_closing(bin)
        dil = binary_erosion(bin, selem=disk(2))
        return 1.0 - dil

    def extract_digits(self, frame_o):
        frame_o = img_as_float(frame_o)
        frame_o = rgb2grey(frame_o)
        frame = rescale(frame_o, 0.5)
        gray = rgb2grey(frame)
        if self.IMSHOW_DBG:
            cv2.imshow('gray', gray)

        dil = self.preprocess_image(gray)
        if self.IMSHOW_DBG:
            cv2.imshow('dil', dil)

        rects = self.rotrects_from_image(dil, 2, True)

        #print('*****')
        cropped = [c if c is not None else
                None for c in [self.crop_rotrect(rect, frame_o) for rect in rects]]
        candidates = []
        cropped_with_indexes = [(c, i) for i, c in enumerate(cropped) if c is not None]
        for c, i in cropped_with_indexes:
            candidates.append(DigitCandidate(rects[i], c))

        return candidates

    # tracking
    def get_roi(self, old_rotrect, image_size):
        w, h = image_size
        center = old_rotrect[0]
        rx = center[0] - MAGICNA_PATKA / 2
        ry = center[1] - MAGICNA_PATKA / 2
        rw = MAGICNA_PATKA
        rh = MAGICNA_PATKA

        if rx + rw > w:
            rw = w - rx
        if rx < 0:
            rx = 0
        if ry + rh > h:
            rh = h - ry
        if ry < 0:
            ry = 0

        return (int(rx), int(ry), int(rw), int(rh))

    def track_digits(self, old_candidates, new_frame):
        if self.tracked:
            return old_candidates

        #self.tracked = True

        frame_o = img_as_float(new_frame)
        gray = rgb2grey(frame_o)

        new_candidates = []

        i=0
        for c in old_candidates:
            h, w = c.image.shape
            if h < 10 or w < 10:
                new_candidates.append(c)
                continue

            rx, ry, rw, rh = self.get_roi(c.rect, (new_frame.shape[1],
                new_frame.shape[0]))
            roi = gray[ry:ry+rh, rx:rx+rw]
            if min(roi.shape) <= 0:
                new_candidates.append(c)
                continue

            roi = self.preprocess_roi(roi)
            if i==0:
                cv2.imshow('roi',img_as_ubyte(roi))
            i+=1
            rects = self.rotrects_from_image(roi, 1, False)
            cands = []
            for rect in rects:
                transl_rect = ((rect[0][0] + rx, rect[0][1] + ry), rect[1],
                        rect[2])
                cands.append(transl_rect)

            #cropped = [(c, rect) for c, rect in [(resize(self.crop_rotrect(rect,
            #    new_frame), (h, w)), rect) for rect in rects] if c is not None]

            if len(cands) > 0:
                #best_i = max(range(len(cropped)), key=lambda i: compare_ssim(c.image,
                #    cropped[i][0], win_size=win_size))
                filt_cands = []
                for r in cands:
                    dw = abs(r[1][0] - c.dimensions[0])
                    dh = abs(r[1][1] - c.dimensions[1])
                    ds = math.sqrt(dw*dw + dh*dh)

                    dx = abs(r[0][0] - c.rect[0][0])
                    dy = abs(r[0][1] - c.rect[0][1])
                    dp = math.sqrt(dx*dx + dy*dy)

                    da = abs(r[2] - c.rect[2])

                    #print('r=',r,'cr=',c.rect,'cd=',c.dimensions)
                    #print('ds=',ds,'dp=',dp,'da=',da)
                    if ds < 30 and dp < 50:
                        filt_cands.append(r)

                best_score = -1
                if not filt_cands:
                    #print("no filt cands")
                    new_candidates.append(c)
                    continue

                for r in filt_cands:
                    cropped = self.crop_rotrect(r, gray)
                    if cropped is None or min(cropped.shape) <= 5:
                        continue

                    cropped = resize(cropped, (h, w))
                    score = compare_ssim(cropped, c.image)
                    if score > best_score:
                        best_score = score
                        best_rect = r
                        best_image = cropped

                if best_score < 0:
                    #print("no valid filt cands")
                    new_candidates.append(c)
                else:
                    new_candidates.append(DigitCandidate(best_rect, best_image))

                #new_candidates.append(c)
            else:
                new_candidates.append(c)
            #print('---')

        #print('***')
        return new_candidates
