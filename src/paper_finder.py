import math

import cv2
import numpy as np

from geometry import ang_diff, cross, line_intersection, undirected_ang_diff


class PaperFinder:
    def __init__(self, target_patience):
        self.target_patience = target_patience
        self.patience = 0
        self.print_debug = False
        pass

    # Checks if ABCD form a rectangle
    def check_sides(self, A, B, C, D):
        side_a1 = (cross(A[0], A[1], A[2], A[3],
                         (C[0]+C[2])//2, (C[1]+C[3])//2) >= 0)
        side_a2 = (cross(A[0], A[1], A[2], A[3],
                         (D[0]+D[2])//2, (D[1]+D[3])//2) >= 0)

        side_b1 = (cross(B[0], B[1], B[2], B[3],
                         (C[0]+C[2])//2, (C[1]+C[3])//2) >= 0)
        side_b2 = (cross(B[0], B[1], B[2], B[3],
                         (D[0]+D[2])//2, (D[1]+D[3])//2) >= 0)

        if side_a1 != side_a2 or side_b1 != side_b2 or side_a1 == side_b1:
            return False

        return True

    # Orients A and B properly
    def fix_dirs(self, A, B):
        sw_ang_a = math.atan2(A[3] - A[1], A[2] - A[0])
        sw_ang_b = math.atan2(B[3] - B[1], B[2] - B[0])
        diff = ang_diff(sw_ang_a, sw_ang_b)
        if diff > np.pi / 2:
            return (B[2], B[3], B[0], B[1]), A
        return A, B

    # Filters very similar lines
    def filter_lines(self, lines, shape):
        new_lines = []
        total = 0
        filtered = 0
        for line in lines:
            total += 1
            xa, ya, xb, yb = line[0]

            D = 5

            if xa < D or xb < D or xa > shape[1] - D or xb > shape[1] - D:
                continue

            if ya < D or yb < D or ya > shape[1] - D or yb > shape[1] - D:
                continue

            A = np.asarray([xa, ya])
            B = np.asarray([xb, yb])
            ang = math.atan2(ya-yb, xa-xb)
            same = False

            # Try to find same line
            for line2, ang2 in new_lines:
                xc, yc, xd, yd = line2
                C = np.asarray([xc, yc])
                D = np.asarray([xd, yd])

                # Get  angular difference
                diff = undirected_ang_diff(ang, ang2)

                # Get distance
                dist_c = np.linalg.norm(np.cross(B-A, A-C))/np.linalg.norm(B-A)
                dist_d = np.linalg.norm(np.cross(B-A, A-D))/np.linalg.norm(B-A)
                dist = max(dist_c, dist_d)

                # Discard if both are very small
                if dist < 20 and diff < 0.1:
                    same = True
                    break

            # Add to output list
            if not same:
                filtered += 1
                new_lines.append(((xa, ya, xb, yb), ang))

        return new_lines

    # Find four lines that represent paper boundary
    def find_abcd(self, lines):
        # Build diffs array
        diffs = []
        n = len(lines)
        for i in range(n):
            for j in range(i+1, n):
                pts_a, ang_a = lines[i]
                pts_b, ang_b = lines[j]
                diff = undirected_ang_diff(ang_a, ang_b)
                diffs.append((diff, i, j))
        diffs.sort(key=lambda tup: tup[0])

        # Get A and B
        A, ang_a = lines[diffs[0][1]]
        B, ang_b = lines[diffs[0][2]]

        # Find C and D
        found = False
        for i in range(1, len(diffs)):
            if diffs[i][1] != diffs[0][1] and \
               diffs[i][1] != diffs[0][2] and \
               diffs[i][2] != diffs[0][1] and \
               diffs[i][2] != diffs[0][2]:

                C, ang_c = lines[diffs[i][1]]
                D, ang_d = lines[diffs[i][2]]

                second_diff = diffs[i][0]
                cross_diff = undirected_ang_diff(ang_a, ang_c)
                found = True
                break

        # Verify we found all four
        if not found:
            if self.print_debug:
                print('PaperFinder: ABCD not found')
            return False, None

        # Verify second diff is not too big
        if second_diff > 0.5:
            if self.print_debug:
                print('PaperFinder: Second diff too big')
            return False, None

        # Verify cross diff has a sensible value
        if abs(cross_diff - np.pi/2) > 0.5:
            if self.print_debug:
                print('PaperFinder: Cross diff looks wrong')
            return False, None

        return True, (A, B, C, D)

    # Find paper
    def find(self, frame):
        # Copy for later crop
        frame_clean = frame.copy()

        # Find lines
        edges = cv2.Canny(frame, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40,
                                minLineLength=95, maxLineGap=10)

        # Return if there are no lines
        if lines is None:
            self.patience = 0
            if self.print_debug:
                print("PaperFinder: Lines is none")
            return False, None

        if self.print_debug:
            print('PaperFinder: {} lines'.format(len(lines)))

        # Something is wrong if there are too many lines, improves FPS
        if len(lines) > 100:
            if self.print_debug:
                print("PaperFinder: Too many lines tbh")
            return False, None

        # Keep only relevant lines
        new_lines = self.filter_lines(lines, frame.shape)

        # There should be at least four
        if (len(new_lines) < 4):
            self.patience = 0
            if self.print_debug:
                print("PaperFinder: No 4 lines")
            return False, None

        # Find (A, B, C, D)
        status, tup = self.find_abcd(new_lines)
        if not status:
            self.patience = 0
            return False, None
        A, B, C, D = tup

        # Fix dirs, make them aligned
        A, B = self.fix_dirs(A, B)
        C, D = self.fix_dirs(C, D)

        # Vertical: A/B, Horizontal: C/D
        diffx_a = abs(A[0] - A[2])
        diffx_c = abs(C[0] - C[2])
        # diffx smaller => vertical
        if diffx_c < diffx_a:
            A, B, C, D = C, D, A, B

        # A should be left of B
        avgx_a = (A[0] + A[2]) / 2
        avgx_b = (B[0] + B[2]) / 2
        if avgx_b < avgx_a:
            A, B = B, A

        # C should be above D
        avgy_c = (C[1] + C[3]) / 2
        avgy_d = (D[1] + D[3]) / 2
        if avgy_d < avgy_c:
            C, D = D, C

        # A and B should have first point up
        if A[1] > A[3]:
            A = (A[2], A[3], A[0], A[1])
            B = (B[2], B[3], B[0], B[1])

        # C and D should have first point left
        if C[0] > C[2]:
            C = (C[2], C[3], C[0], C[1])
            D = (D[2], D[3], D[0], D[1])

        # Check cross products, ABCD should form a rectangle
        if not self.check_sides(A, B, C, D) or not self.check_sides(C, D, A, B):
            self.patience = 0
            if self.print_debug:
                print("PaperFinder: Check sides")
            return False, None

        # Find line intersections
        (height, width) = frame.shape[:2]
        top_left = line_intersection(A, C)
        top_right = line_intersection(B, C)
        bot_right = line_intersection(B, D)
        bot_left = line_intersection(A, D)
        pts = [top_left, top_right, bot_right, bot_left]
        for pt in pts:
            if pt[0] < 0 or pt[0] > width or pt[1] < 0 or pt[1] > height:
                self.patience = 0
                if self.print_debug:
                    print("PaperFinder: At least one intersection is outside bounds")
                return False, None

        # This seems to be the paper, increase patience
        if self.patience < self.target_patience:
            self.patience += 1
            if self.print_debug:
                print("PaperFinder: Patience on {}".format(self.patience))
            return False, None

        # Found it!

        # Draw points, the box and lines
        for pt in pts:
            cv2.circle(frame, (pt[0], pt[1]), 5, (0, 0, 0), 5)

        cv2.drawContours(frame, [np.stack(pts)], 0, (0, 0, 0), 1)

        cv2.circle(frame, (A[0], A[1]), 5, (0, 0, 255), 3)
        cv2.line(frame, (A[0], A[1]), (A[2], A[3]), (0, 0, 255), 1)

        cv2.circle(frame, (B[0], B[1]), 5, (255, 255, 255), 3)
        cv2.line(frame, (B[0], B[1]), (B[2], B[3]), (0, 0, 255), 1)

        cv2.circle(frame, (C[0], C[1]), 5, (255, 0, 0), 3)
        cv2.line(frame, (C[0], C[1]), (C[2], C[3]), (255, 0, 0), 1)

        cv2.circle(frame, (D[0], D[1]), 5, (255, 255, 255), 3)
        cv2.line(frame, (D[0], D[1]), (D[2], D[3]), (255, 0, 0), 1)

        # Find the homography that transforms the paper
        tgt = [(0, 0), (width, 0), (width, height), (0, height)]
        h_mat, status = cv2.findHomography(np.asarray(pts), np.asarray(tgt))

        paper_homo = cv2.warpPerspective(frame_clean, h_mat, (width, height))

        # Crop 5% of the paper to reduce noise and fingers
        ratio = 0.05
        (height, width) = paper_homo.shape[:2]
        cut_h = int(round(ratio * height))
        cut_w = int(round(ratio * width))
        TL = (cut_w, cut_h)
        BR = (width - cut_w, height - cut_h)

        paper_cropped = paper_homo[TL[1]:BR[1], TL[0]:BR[0]]

        # Also return inverse transformation and translation vector
        _, M = cv2.invert(h_mat)
        return True, (paper_cropped, M, TL)
