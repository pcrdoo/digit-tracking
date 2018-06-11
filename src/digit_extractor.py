import math
from math import pow

import cv2
import numpy as np
from skimage import img_as_float, img_as_ubyte
from skimage.color import rgb2grey
from skimage.exposure import equalize_hist
from skimage.filters import threshold_sauvola
from skimage.measure import compare_ssim, label
from skimage.morphology import binary_closing, binary_erosion, disk
from skimage.transform import rescale, resize


# Constants
MIN_SIDE_RATIO = 0.20
MAX_SIDE_RATIO = 1.30
MIN_AREA = 14 * 14
MIN_PIXELS = 50
BIG_BOX_AREA_FRACTION = 0.3
MAGIC = 200

MAX_SIZE_DIFF_FACTOR = 0.5
MAX_POS_DIFF = 20
POS_WEIGHT_ALPHA = 1.5


class DigitCandidate:
    def __init__(self, rect, image, reason=None):
        self.rect = rect
        self.image = image
        self.reason = reason

        self.guess = -1
        self.conf = np.zeros(10)
        self.sharpness = 0
        self.dimensions = rect[1]
        self.orig_image = image
        self.lost_for = 0

    def __repr__(self):
        return "{R = " + repr(self.rect) + "}"


class DigitExtractor:
    def __init__(self, show_debug_images):
        self.show_debug_images = show_debug_images

        # Initial sauvola params
        self._k = 0.45
        self._r = 1.02
        self._ws = 11

        self.tracked = False

    # Use several heuristics to discard rects that are unlikely to have digits
    def rect_eligible(self, rect, sz, area):
        w, h = rect[1]
        if w > h:
            w, h = h, w

        if w == 0 or h == 0:
            return False

        if w < 4 or h < 4:
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

    # Rotate the rect that contains the digit and crop from the original image
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

        # Warp
        new_center = (rect_x - x, rect_y - y)
        new_c_x, new_c_y = new_center
        M = cv2.getRotationMatrix2D(new_center, ang_rot, 1)
        roi = cv2.warpAffine(roi, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # Rotate bounding box
        box = cv2.boxPoints((new_center, size, angle))
        pts = np.int0(cv2.transform(np.array([box]), M))[0]
        pts[pts < 0] = 0

        # Crop
        first_row = np.min(pts[:, 1])
        last_row = np.max(pts[:, 1])
        first_col = np.min(pts[:, 0])
        last_col = np.max(pts[:, 0])
        img_crop = roi[first_row:last_row, first_col:last_col]
        return img_crop

    # Extract all rects that could contain digits from an image
    def rotrects_from_image(self, bin, scale=1, filter=True, area=None):
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
        if not area:
            area = bin.shape[0] * bin.shape[1]

        for c in components:
            sz = c[0].size
            t = np.dstack(c)[0] * scale
            rect = cv2.minAreaRect(t)
            if not filter or self.rect_eligible(rect, sz, area):
                rects.append(rect)

        return rects

    # Preprocessing

    def preprocess_image(self, gray):
        gray = equalize_hist(gray)

        if self.show_debug_images:
            cv2.imshow('gray2', gray)

        thresh = threshold_sauvola(
            gray, window_size=self._ws, k=self._k, r=self._r)
        bin = gray > thresh
        dil = binary_closing(bin)
        dil = binary_erosion(bin, selem=disk(2))
        dil = 1.0 - dil

        return dil

    def preprocess_roi(self, roi):
        thresh = threshold_sauvola(roi, window_size=11, k=0.09)
        bin = roi > thresh
        dil = binary_closing(bin)
        dil = binary_erosion(bin, selem=disk(2))
        return 1.0 - dil

    # Main function: returns a list of DigitCandidates
    def extract_digits(self, frame_o):
        # Prepare image
        frame_o = img_as_float(frame_o)
        frame_o = rgb2grey(frame_o)
        frame = rescale(frame_o, 0.5)
        gray = frame
        if self.show_debug_images:
            cv2.imshow('gray', gray)
        dil = self.preprocess_image(gray)
        if self.show_debug_images:
            cv2.imshow('dil', dil)

        # Find all rects
        rects = self.rotrects_from_image(dil, 2, True)

        # Crop and return candidate list
        cropped = [c if c is not None else
                   None for c in [self.crop_rotrect(rect, frame_o) for rect in rects]]
        candidates = []
        cropped_with_indexes = [(c, i)
                                for i, c in enumerate(cropped) if c is not None]
        for c, i in cropped_with_indexes:
            candidates.append(DigitCandidate(rects[i], c))
        return candidates

    # Tracking

    # Fix roi
    def get_roi(self, old_rotrect, image_size):
        w, h = image_size
        center = old_rotrect[0]
        rx = center[0] - MAGIC / 2
        ry = center[1] - MAGIC / 2
        rw = MAGIC
        rh = MAGIC

        if rx + rw > w:
            rw = w - rx
        if rx < 0:
            rx = 0
        if ry + rh > h:
            rh = h - ry
        if ry < 0:
            ry = 0

        return (int(rx), int(ry), int(rw), int(rh))

    # Tries to locate old_candidates in a new frame and returns their new positions
    def track_digits(self, old_candidates, new_frame, show_idx):
        # Prepare the new frame
        frame_o = img_as_float(new_frame)
        gray_o = rgb2grey(frame_o)
        gray = equalize_hist(gray_o)

        new_candidates = []

        i = 0
        roi_img = self.preprocess_roi(rescale(gray, 0.5))

        # Try to locate each candidate
        for c in old_candidates:
            # Extract the area where the candidate can now be
            i += 1
            h, w = c.image.shape
            if h < 10 or w < 10:
                new_candidates.append(c)
                c.lost_for += 1
                continue

            rx, ry, rw, rh = self.get_roi(c.rect, (new_frame.shape[1],
                                                   new_frame.shape[0]))
            rx = int(rx / 2)
            ry = int(ry / 2)
            rw = int(rw / 2)
            rh = int(rh / 2)

            roi = roi_img[ry:ry+rh, rx:rx+rw]
            if min(roi.shape) <= 0:
                new_candidates.append(c)
                c.lost_for += 1
                continue

            if self.show_debug_images and i-1 == show_idx:
                cv2.imshow('roi', img_as_ubyte(roi))

            # Get digit rects in roi
            rects = self.rotrects_from_image(
                roi, 2, True, frame_o.shape[0] * frame_o.shape[1])
            cands = []
            for rect in rects:
                transl_rect = ((rect[0][0] + 2*rx, rect[0][1] + 2*ry), rect[1],
                               rect[2])
                cands.append(transl_rect)

            if len(cands) == 0:
                # If there are no candidates we lost the digit
                c.lost_for += 1
                new_candidates.append(c)
            else:
                # If there are several candidates we need to choose the best one
                # Calculate metrics
                filt_cands = []
                for r in cands:
                    # Dimensions difference
                    dw = abs(max(r[1]) - max(c.dimensions))
                    dh = abs(min(r[1]) - min(c.dimensions))
                    ds = math.sqrt(dw*dw + dh*dh)

                    # Eucledian distance
                    dx = abs(r[0][0] - c.rect[0][0])
                    dy = abs(r[0][1] - c.rect[0][1])
                    dp = math.sqrt(dx*dx + dy*dy)

                    # Angular difference, unused
                    # da = abs(r[2] - c.rect[2])

                    # Filter based on heuristics
                    if ds < MAX_SIZE_DIFF_FACTOR * np.average(c.dimensions) and dp < MAX_POS_DIFF:
                        filt_cands.append((r, dp))

                # If there are no candidates now we lost the digit
                best_score = -1
                if not filt_cands:
                    new_candidates.append(c)
                    c.lost_for += 1
                    continue

                # Try to find the best one
                max_dp = max([dp for _, dp in filt_cands]) + 1
                for r, dp in filt_cands:
                    cropped = self.crop_rotrect(r, gray_o)
                    if cropped is None or min(cropped.shape) <= 5:
                        continue

                    # Crop, calculate structural similarity
                    cropped = resize(cropped, (h, w))
                    ssim = compare_ssim(cropped, c.orig_image)

                    # Create an aggregated score
                    score = (1.0 - pow(dp / max_dp, POS_WEIGHT_ALPHA)) * ssim
                    if score > best_score:
                        best_score = score
                        best_rect = r
                        best_image = cropped

                # Choose the candidate with the best score if there is one
                if best_score < 0:
                    new_candidates.append(c)
                    c.lost_for += 1
                else:
                    cand = DigitCandidate(best_rect, best_image)
                    cand.dimensions = c.dimensions
                    cand.conf = c.conf
                    cand.guess = c.guess
                    new_candidates.append(cand)
        return new_candidates
