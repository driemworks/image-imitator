import numpy as np
import cv2

# given the input image, set its rgba channels
# to the input params
def manipulate_rgba(img, r, g, b, a):
	1+1

# flip the image along the y-axis
def flip_horizontal(img):
	1+1

# flip the image along the x-axis
def flip_vertical(img):
	1+1

# adjust the gamme of the input image
def adjust_gamma(img, gamma=1.0):
	invGamma = 1.0/gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
					  for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(img, table)