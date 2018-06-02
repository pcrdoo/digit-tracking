import cv2
import numpy
from scipy.signal import find_peaks_cwt
from scipy.signal import savgol_filter


path = raw_input()
img = cv2.imread(path, 0)

def read_img(path):
	global img
	img = cv2.imread(path, 0)
	#img = cv2.medianBlur(img, 5)

def grey_img():
	global img
	img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 29)

	img = 255 - img # reverse color

	kernel = numpy.ones((3, 3), numpy.uint8)
	#img = cv2.erode(img, kernel, iterations = 1) # denoising
	img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def outline_rect():
	global img
	points = cv2.findNonZero(img)
	box = cv2.minAreaRect(points)
	return box

def rotate_img(box):
	global img
	angle = box[2]

	bb = cv2.boxPoints(box)
	center = (int(sum(bb[i][0] for i in range(4))/4), int(sum(bb[i][1] for i in range(4))/4))

	if angle < -45:
		angle += 90
	rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

	rows,cols = img.shape
	img = cv2.warpAffine(img, rotation_matrix, (cols, rows))

def count_rows():
	global img
	rows = cv2.reduce(img, 1, cv2.REDUCE_AVG)

	rows_1d = [rows[i][0] for i in range(len(rows))]

	rows_len = len(rows)

	window = 2*int(len(rows_1d)/100)+1
	other = 7

	
	dist = 8
	MGIJA = 54

	rfiltered = [sum(rows_1d[i - dist: i + dist]) / (dist * 2) for i in range(dist, len(rows_1d) - dist)]

	kurac = [sum(rfiltered[i - dist: i + dist]) / (dist * 2) for i in range(dist, len(rfiltered) - dist)]

	rfiltered = kurac
	#savgol_filter(rows_1d, window, other)#wiener(rows_1d, 30, 0)

	kurac = []

	for i in range(len(rfiltered)-1):
		if rfiltered[i] != rfiltered[i+1]:
			kurac.append(rfiltered[i])

	rfiltered = kurac

	peaks = []
	pd = 1

	for i in range(pd, len(rfiltered) - pd):
		if rfiltered[i] > rfiltered[i-pd] and rfiltered[i] > rfiltered[i+pd]:
			peaks.append(i)
	#peaks = find_peaks_cwt(rfiltered, numpy.arange(1, len(rows)/MGIJA))

	good_peaks = []

	for peak in peaks:
		if rfiltered[peak] > 0:
			good_peaks.append(peak)

	return len(good_peaks)

def main():
	grey_img()

	box = outline_rect()

	rotate_img(box, )

	solution = count_rows()
	print solution
main()