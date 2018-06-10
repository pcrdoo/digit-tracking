import math
import cv2
import numpy as np
from geometry import dist, cross, ang_diff, undirected_ang_diff, line_intersection
from skimage import img_as_float,img_as_ubyte
from skimage.color import label2rgb, rgb2grey, grey2rgb
from skimage.filters import threshold_sauvola

# DEBUG?
DBG = False

class PaperFinder:
    def __init__(self, target_patience):
        self.target_patience = target_patience
        self.patience = 0
        pass
    
    def check_sides(self, A, B, C, D):
        side_a1 = (cross(A[0], A[1], A[2], A[3], (C[0]+C[2])//2, (C[1]+C[3])//2) >= 0)
        side_a2 = (cross(A[0], A[1], A[2], A[3], (D[0]+D[2])//2, (D[1]+D[3])//2) >= 0)
        
        side_b1 = (cross(B[0], B[1], B[2], B[3], (C[0]+C[2])//2, (C[1]+C[3])//2) >= 0)
        side_b2 = (cross(B[0], B[1], B[2], B[3], (D[0]+D[2])//2, (D[1]+D[3])//2) >= 0)

        if side_a1 != side_a2 or side_b1 != side_b2 or side_a1 == side_b1:
            return False
        
        return True

    def fix_dirs(self, A, B):
        sw_ang_a = math.atan2(A[3] - A[1], A[2] - A[0])
        sw_ang_b = math.atan2(B[3] - B[1], B[2] - B[0])
        diff = ang_diff(sw_ang_a, sw_ang_b)
        if diff > np.pi / 2:
            return (B[2], B[3], B[0], B[1]), A
        return A, B

    def filter_lines(self, lines, shape):
        new_lines = []
        total = 0
        filtered = 0
        # print(len(lines))
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
                
                # Ang diff?
                diff = undirected_ang_diff(ang, ang2)

                # Dist?
                dist_c = np.linalg.norm(np.cross(B-A, A-C))/np.linalg.norm(B-A)
                dist_d = np.linalg.norm(np.cross(B-A, A-D))/np.linalg.norm(B-A)
                dist = max(dist_c, dist_d)

                # Discard
                if dist < 20 and diff < 0.1:
                    same = True
                    break

            # Add
            if not same:
                filtered += 1
                new_lines.append( ((xa, ya, xb, yb), ang) )
                
        return new_lines

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

        # Sort
        diffs.sort(key=lambda tup: tup[0])
        
        # Get A and B, easy
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
        
        # Is it ok?
        if not found:
            if DBG:
                print('not found')
            return False, None 
        
        if second_diff > 0.5:
            if DBG:
                print('second diff too big')
            return False, None 
        
        if abs(cross_diff - np.pi/2) > 0.5:
            if DBG:
                print('cross diff')
            return False, None
        
        return True, (A, B, C, D)

    def find(self, frame):
        # Copy
        frame_clean = frame.copy()

        # Find lines
        #frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame_bin = cv2.adaptiveThreshold(frame_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 5)
        #frame_bin = cv2.bitwise_not(frame_bin)
        edges = cv2.Canny(frame, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, minLineLength=95, maxLineGap=10)

        #cv2.imshow('fn', frame)
        #cv2.imshow('edges', edges)
        #cv2.imshow('lines', lines)

        # Nope
        if lines is None:
            self.patience = 0
            if DBG:
                print("Lines is none")
            return False, None

        if DBG:
            print(len(lines))
        
        if len(lines) > 100:
            if DBG:
                print("Too many lines tbh")
            return False, None

        # Keep only relevant lines
        new_lines = self.filter_lines(lines, frame.shape)

        # Nope
        if (len(new_lines) < 4):
            self.patience = 0
            if DBG:
                print("No 4 lines")
            return False, None
        
        # Find (A, B, C, D)
        status, tup = self.find_abcd(new_lines)

        # Nope
        if not status:
            self.patience = 0
            if DBG:
                print("Abcd")
            return False, None
        
        # Unpack
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
        
        # Nope
        if not self.check_sides(A, B, C, D) or not self.check_sides(C, D, A, B):
            self.patience = 0
            if DBG:
                print("Check sides")
            return False, None
        
        # HW!
        (height, width) = frame.shape[:2]

        # Intersections
        top_left = line_intersection(A, C)
        top_right = line_intersection(B, C)
        bot_right = line_intersection(B, D)
        bot_left = line_intersection(A, D)
        pts = [top_left, top_right, bot_right, bot_left]
        for pt in pts:
            if pt[0] < 0 or pt[0] > width or pt[1] < 0 or pt[1] > height:
                self.patience = 0
                if DBG:
                    print("Intersection is out")
                return False, None
        
        # Nope
        if self.patience < self.target_patience:
            self.patience += 1
            if DBG:
                print("patience on {}".format(self.patience))
            return False, None

        # FOUND IT! ####################################################

        # Draw points
        for pt in pts:
            cv2.circle(frame, (pt[0], pt[1]), 5, (0, 0, 0), 5)

        # Draw the box
        cv2.drawContours(frame,[np.stack(pts)],0,(0,0,0),1)

        # Draw all lines, A and C are marked with colored dots


        cv2.circle(frame, (A[0], A[1]), 5, (0, 0, 255), 3)
        cv2.line(frame,(A[0],A[1]),(A[2],A[3]),(0,0,255),1)

        cv2.circle(frame, (B[0], B[1]), 5, (255, 255, 255), 3)
        cv2.line(frame,(B[0],B[1]),(B[2],B[3]),(0,0,255),1)

        cv2.circle(frame, (C[0], C[1]), 5, (255, 0, 0), 3)
        cv2.line(frame,(C[0],C[1]),(C[2],C[3]),(255,0,0),1)

        cv2.circle(frame, (D[0], D[1]), 5, (255, 255, 255), 3)
        cv2.line(frame,(D[0],D[1]),(D[2],D[3]),(255,0,0),1)


        ######################################################################33

        # pts = [top_left, top_right, bot_right, bot_left]
        tgt = [(0, 0), (width, 0), (width, height), (0, height)]
        h_mat, status = cv2.findHomography(np.asarray(pts), np.asarray(tgt))

        paper_homo = cv2.warpPerspective(frame_clean, h_mat, (width, height))

        # OLD: box approach
        """
        rect = cv2.minAreaRect(np.asarray(pts))
        box = cv2.boxPoints(rect)
        box = np.int0(np.around(cv2.boxPoints(rect)))
        
        #cv2.drawContours(frame,[box],0,(0,255,0),2)

        ang = -(90 + rect[2]) if rect[2] < -45 else -rect[2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, -ang, 1.0)

        box_rotated = cv2.transform(np.array([box]), matrix)
        frame_rot = cv2.warpAffine(frame_clean, matrix, (width, height),
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        # Crop
        top_left = np.min(np.min(box_rotated, axis=1), axis=0)
        bot_right = np.max(np.max(box_rotated, axis=1), axis=0)
        cropped = frame_rot[top_left[1]:bot_right[1], top_left[0]:bot_right[0]]
        """

        # Crop hands?
        ratio = 0.05 # TODO
        (height, width) = paper_homo.shape[:2]
        cut_h = int(round(ratio * height))
        cut_w = int(round(ratio * width))
        TL = (cut_w, cut_h)
        BR = (width - cut_w, height - cut_h)

        paper_cropped = paper_homo[TL[1]:BR[1], TL[0]:BR[0]]
        #paper_cropped[0:h, 0:w] = 0

        # cv2.imshow('paper', paper_cropped)

        _, h_inv = cv2.invert(h_mat)
        return True, (paper_cropped, h_inv, TL)
