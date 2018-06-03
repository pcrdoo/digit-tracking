import cv2
import numpy as np
import math
from skimage.color import grey2rgb
from skimage import img_as_ubyte, img_as_float

from paper_finder import PaperFinder
from digit_extraction import DigitExtractor
from digit_classifier import DigitClassifier

# Capture
cap = cv2.VideoCapture(0)

skip = 10
nb_frame = 0
patience = 0

# Objects
font = cv2.FONT_HERSHEY_SIMPLEX
show = 0

paper_finder = PaperFinder(target_patience = 5)
clf = DigitClassifier()
ext = DigitExtractor()

while True:
    # Capture frame-by-frame
    nb_frame += 1
    # print("frame {}".format(nb_frame))
    ret, frame = cap.read()
    cv2.waitKey(1)

    # Skip?
    if skip > 0:
        skip -= 1
        cv2.imshow("frame", frame)
        continue

    status, info = paper_finder.find(frame)
    if not status:
       # print("not find")
        cv2.imshow("frame", frame)
        continue
    
    # Found paper, show
    paper, invmat, trans = info
    cv2.imshow("frame", frame)
    cv2.imshow("paper", paper)

    # Block
    while not(cv2.waitKey(1) & 0xFF == ord('p')):
        pass
    break

# Found paper do other shit here
print(paper.shape)

paper_float = img_as_float(paper)
candidates = ext.extract_digits(paper, clf.img_cols, clf.img_rows)
all_imgs = np.array([c.image for c in candidates])
confidences = clf.predict(all_imgs)

print(len(candidates))

if show == 0:
        result_s = paper_float
        for i, cand in enumerate(candidates):
            print(i)
            rect = cand.rect
            image = cand.image

            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(paper_float,[box],0,(0,0,1),2)

            x_off = int(rect[0][0] - image.shape[1] / 2)
            y_off = int(rect[0][1] - image.shape[0] / 2)
            try:
                paper_float[y_off:y_off+image.shape[0], x_off:x_off+image.shape[1]] = grey2rgb(image)
            except ValueError:
                pass

            max_i = max(range(10), key=lambda j: confidences[i][j])
            max_c = confidences[i][max_i]

            cv2.putText(paper_float,str(max_i),(int(rect[0][0]) - 5, int(rect[0][1]) - 20), font, 0.6,(1,0,0),2,cv2.LINE_AA)
elif show == 1:
    result_s = paper_float

# show result?
cv2.imshow('result', result_s)
while not(cv2.waitKey(1) & 0xFF == ord('p')):
    pass

# show result on final pic
height, width, channels = frame.shape
trans2 = (height-trans[1]-result_s.shape[0], width-trans[0]-result_s.shape[1])
np.pad(result_s, ((trans[1], trans[0]), (trans2[1], trans2[0]), (0, 0)), 'constant')

# now result padded
cv2.imshow('result', img_as_ubyte(result_s))
while not(cv2.waitKey(1) & 0xFF == ord('p')):
    pass

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
