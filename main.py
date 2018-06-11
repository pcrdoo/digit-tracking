import cv2
import numpy as np
import math
from skimage.color import grey2rgb
from skimage import img_as_ubyte, img_as_float
from skimage.exposure import equalize_hist

from paper_finder import PaperFinder
from digit_extraction import DigitExtractor
from digit_classifier import DigitClassifier
from utils import transform_img
from time import sleep

# IMSHOW DBG
IMSHOW_DBG = False

# Capture
cap = cv2.VideoCapture(0)

skip = 10
nb_frame = 0
patience = 0

mnist_img_width = 28
mnist_img_height = 28

# Objects
paper_finder = PaperFinder(target_patience = 5)
paper = None
clf = DigitClassifier(mnist_img_height, mnist_img_width)
ext = DigitExtractor(IMSHOW_DBG)

def inverse_transform(rect, M, TL):
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

    box = np.int0(box)
    return cv2.minAreaRect(box)

def draw_candidate(target, rect, image, guess, confs, M = None, TL = None, reason = None, pretty = False):
    font = cv2.FONT_HERSHEY_SIMPLEX

    # center and box are used later for positioning
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

    # actual bounding box?
    x,y,w,h = cv2.boundingRect(box)

    #print('{} and {} and {} and {}'.format(x, y, x + w, y + h))
    #print('sz = {} and {}'.format(target.shape[0], target.shape[1]))

    # done, draw
    if not pretty:
        cv2.drawContours(target,[box],0,(0,0,1),2)
        
        x_off = int(center[0] - image.shape[1] / 2)
        y_off = int(center[1] - image.shape[0] / 2)
        try:
            target[y_off:y_off+image.shape[0], x_off:x_off+image.shape[1]] = grey2rgb(image)
        except ValueError:
            pass
    else:
        cv2.rectangle(target,(x,y),(x+w,y+h),(255,0,255),1)

    if not pretty:
        cv2.putText(target,str(guess),(int(center[0]) - 5, int(center[1]) - 20), font, 0.6,(1,0,0),2,cv2.LINE_AA)
    else:
        cv2.putText(target,str(guess), (int(x + w/2 - 6), int(y) - 2), font, 0.6, (1,0,1), 2, cv2.LINE_AA)
    
    #cv2.putText(target,reason,(int(center[0]) - 5, int(center[1]) + 20), font, 0.6,(1,0,0),2,cv2.LINE_AA)

while True:
    # Capture frame-by-frame
    nb_frame += 1
    # print("frame {}".format(nb_frame))
    ret, frame = cap.read()
    # print(frame.shape)
    if frame.shape != (480, 640):
        frame = cv2.resize(frame, (640, 480))

    k = chr(cv2.waitKey(1) & 0xFF)
    if k == 'l':
        ext._k += 0.01
        print('k=',ext._k)
    elif k == 'k':
        ext._k -= 0.01
        print('k=',ext._k)
    elif k == 'p':
        ext._ws += 2
        print('ws=',ext._ws)
    elif k == 'o':
        ext._ws -= 2
        print('ws=',ext._ws)
    elif k == 'm':
        ext._r += 0.01
        print('r=',ext._r)
    elif k == 'n':
        ext._r -= 0.01
        print('r=',ext._r)
    elif k == 'q':
        zzz = input()

    # Skip?
    if skip > 0:
        skip -= 1
        cv2.imshow("number-tracking", frame)
        if IMSHOW_DBG:
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
        cv2.imshow("number-tracking", frame)
        continue

    # Found paper, show
    paper, h_inv, TL = info
    if IMSHOW_DBG:
        cv2.imshow("number-tracking", frame)

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
    candidates = ext.extract_digits(paper)
    if not candidates:
        continue

    # Sharpness
    for cand in candidates:
        img = grey2rgb(cand.image)
        laplacian = cv2.Laplacian(cand.image,cv2.CV_64F)
        sharpness = np.max(laplacian)
        cand.sharpness = sharpness

    # Transform images
    transformed = [transform_img(c.image, mnist_img_height, mnist_img_width) for c in candidates]

    #cv2.imshow("first transformed", transformed[0])

    #with open("log.txt", "w") as f:
    #    for t in transformed:
    #        f.write(str(t))
    #        f.write("\n===\n")

    all_imgs = np.array(transformed)
    
    # Get confidences from the model
    filtered_candidates = []
    CONF_THRESH = 0.5

    # Threshold
    confidences = clf.predict(all_imgs)
    for i, cand in enumerate(candidates):
        # set guess/conf and thresh on that
        cand.guess = max(range(10), key=lambda j: confidences[i][j])
        cand.conf = confidences[i][cand.guess]
        if cand.conf <= CONF_THRESH:
            continue
        
        # finger filter: if rect center is in banned area => remove
        center = cand.rect[0]
        c_x, c_y = center
        p_h, p_w, _ = paper.shape
        BAN_STRIP_H = p_h/3
        BAN_STRIP_W = p_w/18
        if c_x < BAN_STRIP_W and abs(p_h/2 - c_y) < BAN_STRIP_H/2:
            continue
        c_x = p_w - c_x
        if c_x < BAN_STRIP_W and abs(p_h/2 - c_y) < BAN_STRIP_H/2:
            continue

        # push
        filtered_candidates.append(cand)
        #print("{} -> {}".format(cand.guess, cand.SH))
    #print("DONE")

    # Move back 
    candidates = filtered_candidates

    # Print confidences
    print_confs = False 
    if print_confs:
        print("===== CONFIDENCES ==== ")
        for i in range(len(confidences)):
            verdict = "Digit {}:".format(i)
            for j in range(10):
                if confidences[i][j] > 0.01:
                    verdict += '{} ({}%) '.format(j, round(confidences[i][j]*100))
            print(verdict)
        print("===== =========== ==== ")

    # Draw candidates
    paper_result = img_as_float(paper)
    for i, cand in enumerate(candidates):
        rect = cand.rect
        image = cand.image
        draw_candidate(paper_result, cand.rect, transformed[i], cand.guess, confidences[i])
    if IMSHOW_DBG:
        cv2.imshow('paper_result', paper_result)

    # Transform back and draw on original frame
    frame_result = img_as_float(frame_clean.copy())
    for i, cand in enumerate(candidates):
        rect = cand.rect
        image = cand.image
        reason = cand.reason
        draw_candidate(frame_result, cand.rect, transformed[i], cand.guess, confidences[i], h_inv, TL, reason, pretty=True)
    if IMSHOW_DBG:
        cv2.imshow('frame_result', frame_result)
    else:
        cv2.imshow('number-tracking', frame_result)

    for c in candidates:
        c.rect = inverse_transform(c.rect, h_inv, TL)
        c.dimensions = c.rect[1]

    break

first_cd = candidates
idx = 0
show_all = True
show_old = False
show_next_frame = True
orig_frame = frame

def draw_rotrect(r, img, col):
    box = cv2.boxPoints(r)
    box = np.int0(box)
    cv2.drawContours(img,[box],0,col,2)

tracked = False
freeze = False

while True:
    if not freeze:
        ret, frame = cap.read()
        candidates = ext.track_digits(candidates, frame, idx)

    #freeze=True
    k = chr(cv2.waitKey(1) & 0xFF)
    if k == 'r':
        idx += 1
        print('Showing cand', idx)
    elif k == 'e':
        idx -= 1
        print('Showing cand', idx)
    elif k == 'a':
        show_all = not show_all
        print('Show all', show_all)
    elif k == 'f':
        show_next_frame = not show_next_frame
        print('Showing next frame', show_next_frame)
    elif k == 'o':
        show_old = not show_old
        print('Showing old cands',show_old)
    elif k == 'x':
        freeze = not freeze

    frame_result = img_as_float(frame.copy()) if show_next_frame else img_as_float(orig_frame.copy())
    if show_all:
        if show_old:
            for c in first_cd:
                draw_rotrect(c.rect, frame_result, (1,0,0))
        for c in candidates:
            draw_rotrect(c.rect, frame_result, (0,0,1))
    else:
        if not (idx >= len(candidates) or idx >= len(first_cd) or idx < 0):
            draw_rotrect(first_cd[idx].rect, frame_result, (1,0,0))
            draw_rotrect(candidates[idx].rect, frame_result, (0,0,1))
            cv2.imshow('oldimg',first_cd[idx].image)
            cv2.imshow('newimg',candidates[idx].image)

    cv2.imshow('frame_result', frame_result)
    sleep(0.1)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
