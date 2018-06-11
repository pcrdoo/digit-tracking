import cv2
import numpy as np
from skimage import img_as_float
from skimage.color import grey2rgb

from digit_classifier import DigitClassifier
from digit_extractor import DigitExtractor
from paper_finder import PaperFinder
from utils import transform_img

# Capture
cap = cv2.VideoCapture(0)
nb_frame = 0

# Params
show_debug_images = False
skip = 10
patience = 0
mnist_img_width = 28
mnist_img_height = 28

# Objects
paper = None
paper_finder = PaperFinder(target_patience=2)
clf = DigitClassifier(mnist_img_height, mnist_img_width)
ext = DigitExtractor(show_debug_images)

# If at least LOST_FACTOR of all candidates have been lost for more than
# LOST_FRAMES frames, reset everything
LOST_FACTOR = 0.5
LOST_FRAMES = 20

font = cv2.FONT_HERSHEY_SIMPLEX

# Do inverse homography


def inverse_transform(rect, M, TL):
    # Center and box are used later for positioning
    center = rect[0]
    box = cv2.boxPoints(rect)

    # Transform
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

    # Return min area rect
    box = np.int0(box)
    return cv2.minAreaRect(box)

# Draw a rotated rect on an image


def draw_rotrect(r, img, col):
    box = cv2.boxPoints(r)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, col, 2)

# Draw a candidate


def draw_candidate(img, cand, confs, M=None, TL=None, pretty=False):

    # Center and box are used later for positioning
    center = cand.rect[0]
    box = cv2.boxPoints(cand.rect)

    # Transform
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
    center = np.int0(center)
    box = np.int0(box)

    # Find actual bounding box
    x, y, w, h = cv2.boundingRect(box)

    # Draw
    if not pretty:
        cv2.drawContours(img, [box], 0, (0, 0, 1), 2)

        x_off = int(center[0] - cand.image.shape[1] / 2)
        y_off = int(center[1] - cand.image.shape[0] / 2)
        try:
            img[y_off:y_off+cand.image.shape[0], x_off:x_off +
                cand.image.shape[1]] = grey2rgb(cand.image)
        except ValueError:
            pass
    else:
        cv2.drawContours(img, [box], 0, (1, 0, 1), 1)

    # Get the guess
    guess = str(cand.guess)
    leet_mode = False
    if leet_mode:
        leet = {'0': 'O', '1': 'L', '2': 'Z', '3': 'E', '4': 'A',
                '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'P'}
        guess = leet[guess]

    # Draw text
    if not pretty:
        cv2.putText(img, guess, (int(
            center[0]) - 5, int(center[1]) - 20), font, 0.6, (1, 0, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(img, guess, (int(x + w/2 - 6),
                                 int(y) - 2), font, 0.6, (1, 0, 1), 2, cv2.LINE_AA)

# For confidence aggregation


def update_candidate_confs(cand, confidences):
    cand.conf += confidences
    cand.guess = max(range(10), key=lambda i: cand.conf[i])


# Main loop
while True:
    # Paper finding loop
    while True:
        # Capture frame-by-frame
        nb_frame += 1
        ret, frame = cap.read()
        if frame.shape != (480, 640):
            frame = cv2.resize(frame, (640, 480))

        # Sauvola tuning
        k = chr(cv2.waitKey(1) & 0xFF)
        if k == 'l':
            ext._k += 0.01
            print('k=', ext._k)
        elif k == 'k':
            ext._k -= 0.01
            print('k=', ext._k)
        elif k == 'p':
            ext._ws += 2
            print('ws=', ext._ws)
        elif k == 'o':
            ext._ws -= 2
            print('ws=', ext._ws)
        elif k == 'm':
            ext._r += 0.01
            print('r=', ext._r)
        elif k == 'n':
            ext._r -= 0.01
            print('r=', ext._r)
        elif k == 'q':
            _ = input()

        # Skip first n frames
        if skip > 0:
            skip -= 1
            cv2.imshow("digit-tracking", frame)
            if show_debug_images:
                if paper is not None:
                    cv2.imshow("paper", paper)
            continue

        # Get frame info
        (height, width) = frame.shape[:2]
        frame_clean = frame.copy()

        # Try to find paper
        status, info = paper_finder.find(frame)
        if not status:
            cv2.imshow("digit-tracking", frame)
            continue
        paper, M, TL = info
        if show_debug_images:
            cv2.imshow("digit-tracking", frame)

        # Extract digits
        candidates = ext.extract_digits(paper)
        if not candidates:
            continue

        # Calculate sharpness
        for cand in candidates:
            img = grey2rgb(cand.image)
            laplacian = cv2.Laplacian(cand.image, cv2.CV_64F)
            sharpness = np.max(laplacian)
            cand.sharpness = sharpness

        # Transform images
        transformed = [transform_img(
            c.image, mnist_img_height, mnist_img_width) for c in candidates]
        all_imgs = np.array(transformed)

        # Get confidences from the model
        filtered_candidates = []
        CONF_THRESH = 0.5

        # Perform thresholding
        confidences = clf.predict(all_imgs)
        for i, cand in enumerate(candidates):

            # Set guess/conf and threshold on that
            update_candidate_confs(cand, confidences[i])
            cand.conf = confidences[i][cand.guess]
            if cand.conf <= CONF_THRESH:
                continue

            # Simple finger filter: if rect center is in a banned area => remove
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

            filtered_candidates.append(cand)
        candidates = filtered_candidates

        # Print confidences
        print_confs = False
        if print_confs:
            print("===== CONFIDENCES ==== ")
            for i in range(len(confidences)):
                verdict = "Digit {}:".format(i)
                for j in range(10):
                    if confidences[i][j] > 0.01:
                        verdict += '{} ({}%) '.format(j,
                                                      round(confidences[i][j]*100))
                print(verdict)
            print("===== =========== ==== ")

        # Draw candidates
        paper_result = img_as_float(paper)
        for i, cand in enumerate(candidates):
            rect = cand.rect
            image = cand.image
            draw_candidate(paper_result, cand, confidences[i])
        if show_debug_images:
            cv2.imshow('paper_result', paper_result)

        # Transform back and draw on original frame
        frame_result = img_as_float(frame_clean.copy())
        for i, cand in enumerate(candidates):
            rect = cand.rect
            image = cand.image
            reason = cand.reason
            draw_candidate(frame_result, cand,
                           confidences[i], M, TL, pretty=True)
        if show_debug_images:
            cv2.imshow('frame_result', frame_result)
        else:
            cv2.imshow('digit-tracking', frame_result)

        # Inverse transform candidates for tracking
        for c in candidates:
            c.rect = inverse_transform(c.rect, M, TL)
            c.dimensions = c.rect[1]

        # Break, start tracking loop
        print("Found paper, now tracking.")
        break

    # Tracking vars
    first_cd = candidates
    idx = 0
    show_all = True
    show_old = False
    show_next_frame = True
    orig_frame = frame
    tracked = False
    freeze = False
    debug_tracking = False

    # Tracking loop
    while True:
        if not freeze:
            ret, frame = cap.read()
            candidates = ext.track_digits(candidates, frame, idx)

        # Debug prints
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
            print('Showing old cands', show_old)
        elif k == 'x':
            freeze = not freeze
        elif k == 't':
            debug_tracking = not debug_tracking

        # Count the number of lost digits, if too many give up
        num_lost = len([c for c in candidates if c.lost_for > LOST_FRAMES])
        if num_lost > LOST_FACTOR * len(candidates):
            print("Lost paper. Trying from the start.")
            break

        # More debug info
        if debug_tracking:
            frame_result = img_as_float(
                frame.copy()) if show_next_frame else img_as_float(orig_frame.copy())
            if show_all:
                if show_old:
                    for c in first_cd:
                        draw_rotrect(c.rect, frame_result, (1, 0, 0))
                for c in candidates:
                    draw_rotrect(c.rect, frame_result, (0, 0, 1))
            else:
                if not (idx >= len(candidates) or idx >= len(first_cd) or idx < 0):
                    draw_rotrect(first_cd[idx].rect, frame_result, (1, 0, 0))
                    draw_rotrect(candidates[idx].rect, frame_result, (0, 0, 1))
                    cv2.imshow('oldimg', first_cd[idx].image)
                    cv2.imshow('newimg', candidates[idx].image)

            cv2.imshow('frame_result', frame_result)

        # Draw on original frame
        frame_result = img_as_float(frame.copy())
        for i, cand in enumerate(candidates):
            rect = cand.rect
            image = cand.image
            reason = cand.reason
            draw_candidate(frame_result, cand, confidences[i], pretty=True)

        # Show the final image
        cv2.imshow('digit-tracking', frame_result)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
