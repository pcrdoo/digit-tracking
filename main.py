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
paper_finder = PaperFinder(target_patience = 5)
paper = None
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
        if paper is not None:
            cv2.imshow("paper", paper)
        continue

    # Frame info
    (height, width) = frame.shape[:2]
    frame_clean = frame.copy()

    # Find paper
    status, info = paper_finder.find(frame)
    if not status:
       # print("not find")
        cv2.imshow("frame", frame)
        if paper is not None:
            cv2.imshow("paper", paper)
        continue
    
    # Found paper, show
    paper, h_inv, TL = info
    cv2.imshow("frame", frame)
    cv2.imshow("paper", paper)

    """
    paper_uncrop = np.pad(paper, 
                          ((TL[1], TL[1]), 
                           (TL[0], TL[0]), 
                           (0,0)), 
                          'constant', 
                          constant_values = ((128,)))
    cv2.imshow("paper_uncrop", paper_uncrop)
    """

    # Extract digits
    candidates = ext.extract_digits(paper, clf.img_cols, clf.img_rows)
    all_imgs = np.array([c.image for c in candidates])

    # Get confidences from the model
    confidences = clf.predict(all_imgs)

    # Draw candidates
    paper_result = img_as_float(paper)
    for i, cand in enumerate(candidates):
        rect = cand.rect
        image = cand.image
        ext.draw_candidate(paper_result, cand.rect, cand.image, confidences[i])
    cv2.imshow('paper_result', paper_result)

    # Transform back and draw on original frame
    frame_result = img_as_float(frame_clean.copy())
    for i, cand in enumerate(candidates):
        rect = cand.rect
        image = cand.image
        ext.draw_candidate(frame_result, cand.rect, cand.image, confidences[i], h_inv, TL)
    cv2.imshow('frame_result', frame_result)

    # First time you find a paper target patience is one
    # TODO: we should actually 'break' here and start doing
    # incremental bullshit
    paper_finder.target_patience = 1
    continue

    # Block
    while not(cv2.waitKey(1) & 0xFF == ord('p')):
        pass
    break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
