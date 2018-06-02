import cv2
import numpy as np
import math

from paper_finder import PaperFinder

# Capture
cap = cv2.VideoCapture(0)

skip = 10
nb_frame = 0
patience = 0

# Objects
paper_finder = PaperFinder(target_patience = 5)

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

    status, paper = paper_finder.find(frame)
    if not status:
        # print("not find")
        cv2.imshow("frame", frame)
        continue
    
    # Found paper, do other shit here
    cv2.imshow("frame", frame)
    cv2.imshow("paper", paper)

    # Block
    while not(cv2.waitKey(1) & 0xFF == ord('p')):
        pass

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
