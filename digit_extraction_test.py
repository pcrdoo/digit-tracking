import cv2
from digit_extraction import DigitExtractor
from digit_classifier import DigitClassifier
from skimage.color import grey2rgb
from skimage import img_as_ubyte, img_as_float
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)
show = 1
result_s = None
clf = DigitClassifier()
ext = DigitExtractor()

show = 0

while True:
    # Capture frame-by-frame
    ret, frame_o = cap.read()
    frame_float = img_as_float(frame_o)
    candidates = ext.extract_digits(frame_float, clf.img_cols, clf.img_rows)
    all_imgs = np.array([c.image for c in candidates])
    confidences = clf.predict(all_imgs)

    if show == 0:
        result_s = frame_float
        for i, cand in enumerate(candidates):
            rect = cand.rect
            image = cand.image

            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(frame_float,[box],0,(0,0,1),2)

            x_off = int(rect[0][0] - image.shape[1] / 2)
            y_off = int(rect[0][1] - image.shape[0] / 2)
            try:
                frame_float[y_off:y_off+image.shape[0], x_off:x_off+image.shape[1]] = grey2rgb(image)
            except ValueError:
                pass

            max_i = max(range(10), key=lambda j: confidences[i][j])
            max_c = confidences[i][max_i]

            cv2.putText(frame_float,str(max_i),(int(rect[0][0]) - 5, int(rect[0][1]) - 20), font, 0.6,(1,0,0),2,cv2.LINE_AA)
    elif show == 1:
        result_s = frame_float

    # Display the resulting frame
    cv2.imshow('frame', img_as_ubyte(result_s))
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

    if k == ord('s'):
        show = (show + 1) % 2

