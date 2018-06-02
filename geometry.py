import math
import numpy as np

def dist(x1, y1, x2, y2):
    return math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))

def cross(x1, y1, x2, y2, x3, y3):
    return (x2 - x1) * (y3 - y2) - (x3 - x2) * (y2 - y1)

# works on atan2 output
def ang_diff(ang1, ang2):
    if ang1 < 0:
        ang1 += 2 * np.pi
    if ang2 < 0:
        ang2 += 2 * np.pi
    diff = abs(ang1 - ang2)
    if diff > np.pi:
        diff = 2 * np.pi - diff 
    return diff

def undirected_ang_diff(ang1, ang2):
    diff = ang_diff(ang1, ang2)
    return min(diff, np.pi - diff)

def line_intersection(line1, line2):
    line1 = ((line1[0], line1[1]), (line1[2], line1[3]))
    line2 = ((line2[0], line2[1]), (line2[2], line2[3]))

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(round(x)), int(round(y))