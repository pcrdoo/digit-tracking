import cv2
from skimage.filters import threshold_sauvola, threshold_otsu
from skimage import img_as_ubyte, img_as_float
from skimage.morphology import binary_erosion, binary_closing, binary_opening, disk
from skimage.measure import label, regionprops
from skimage.color import label2rgb, rgb2grey, grey2rgb
from skimage.transform import rescale
from math import sqrt
import numpy as np

# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

MIN_SIDE_RATIO = 0.20
MAX_SIDE_RATIO = 1.30
MIN_AREA = 14 * 14
MIN_PIXELS = 50
BIG_BOX_AREA_FRACTION = 0.3

cap = cv2.VideoCapture(0)
show_bin = False
result_s = None

def rect_eligible(rect, sz, area):
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

def crop_rotrect(rect, img):
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
    return binary_erosion(crop > thresh)

while True:
    # Capture frame-by-frame
    ret, frame_o = cap.read()
    frame_o = img_as_float(frame_o)
    frame_color = frame_o
    frame_o = rgb2grey(frame_o)
    frame = rescale(frame_o, 0.5)

    # Our operations on the frame come here
    gray = rgb2grey(frame)
    thresh = threshold_sauvola(gray, window_size=17, k=0.04, r=0.2)
    bin = gray > thresh
    dil = binary_closing(bin)
    dil = binary_erosion(bin, selem=disk(2))
    dil = 1.0 - dil
    lab, num = label(dil, return_num=True)

    if not num:
        continue

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
        if rect_eligible(rect, sz, area):
            rects.append(rect)

    if show_bin:
        result_s = rescale(dil, 2)
    else:
        result_s = frame_color
        for rect in rects:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            crop = crop_rotrect(rect, frame_o)
            if crop is None:
                continue

            x_off = int(rect[0][0] - crop.shape[1] / 2)
            y_off = int(rect[0][1] - crop.shape[0] / 2)
            try:
                frame_color[y_off:y_off+crop.shape[0], x_off:x_off+crop.shape[1]] = grey2rgb(crop)
            except ValueError:
                pass
            cv2.drawContours(frame_color,[box],0,(0,0,1),2)

    # Display the resulting frame
    cv2.imshow('frame', img_as_ubyte(result_s))
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

    if k == ord('s'):
        show_bin = not show_bin


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
