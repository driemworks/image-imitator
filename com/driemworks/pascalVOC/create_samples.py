from __future__ import print_function

import os
import numpy as np
import cv2
import random
from random import shuffle

from com.driemworks.pascalVOC.pyObjects.annotation import Annotation
from com.driemworks.pascalVOC.pyObjects.annotation_size import Size
from com.driemworks.pascalVOC.pyObjects.BndBox import BoundingBox
from com.driemworks.pascalVOC.config import config
from com.driemworks.pascalVOC.config.config import Labels

folder = "/samples/"

width = 640
height = 480

BLUE = [255, 0, 0]        # rectangle color
RED = [0, 0, 255]         # PR BG
GREEN = [0, 255, 0]       # PR FG
BLACK = [0, 0, 0]         # sure BG
WHITE = [255, 255, 255]   # sure FG

DRAW_BG = {'color': BLACK, 'val': 0}
DRAW_FG = {'color': WHITE, 'val': 1}
DRAW_PR_FG = {'color': GREEN, 'val': 3}
DRAW_PR_BG = {'color': RED, 'val': 2}

bg_dir = config.background_image_directory
img_dir = config.input_image_directory
save_dir = config.save_image_dir
anno_dir = config.save_anno_dir

# setting up flags
rect = (0, 0, 1, 1)
drawing = False         # flag for drawing curves
rectangle = False       # flag for drawing rect
rect_over = False       # flag to check if rect drawn
rect_or_mask = 100      # flag for selecting rect or mask mode
value = DRAW_FG         # drawing initialized to FG
thickness = 3           # brush thickness

X = 0
Y = 0

selected_label = None

def select_label(event):
	try:
		input("press enter to continue...")
	except SyntaxError:
		pass

def onMouseInputImage(event, x, y, flags, param):
	global img,img2,drawing,value,mask,rectangle,rect,rect_or_mask,ix,iy,rect_over, X, Y

	# if selected_label is None:
	# 	selected_label(event)

	# Draw Rectangle
	if event == cv2.EVENT_RBUTTONDOWN:
		rectangle = True
		ix,iy = x,y

	elif event == cv2.EVENT_MOUSEMOVE:
		if rectangle:
			img = img2.copy()
			cv2.rectangle(img, (ix, iy), (x, y), BLUE, 2)
			rect = (min(ix, x), min(iy, y), abs(ix-x), abs(iy-y))
			rect_or_mask = 0

	elif event == cv2.EVENT_RBUTTONUP:
		rectangle = False
		rect_over = True
		X = x
		Y = y
		cv2.rectangle(img,(ix, iy), (x, y), BLUE, 2)
		rect = (min(ix, x), min(iy, y), abs(ix-x), abs(iy-y))
		rect_or_mask = 0

	# draw touchup curves or labels
	if event == cv2.EVENT_LBUTTONDOWN:
		if rect_over == False:
			print("first draw rectangle \n")
		else:
			drawing = True
			cv2.circle(img,(x,y),thickness,value['color'],-1)
			cv2.circle(mask,(x,y),thickness,value['val'],-1)

	elif event == cv2.EVENT_MOUSEMOVE:
		if drawing:
			cv2.circle(img, (x, y), thickness, value['color'], -1)
			cv2.circle(mask, (x, y), thickness, value['val'], -1)

	elif event == cv2.EVENT_LBUTTONUP:
		if drawing:
			drawing = False
			cv2.circle(img,(x, y), thickness,value['color'],-1)
			cv2.circle(mask,(x, y), thickness,value['val'],-1)


# draw labels on output image
def onMouseOutputImage(event, x, y, flags, param):
	global refPt, output, stg_label
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
	elif event == cv2.EVENT_LBUTTONUP:
		refPt.append((x, y))
		staged_rects.append(refPt)
		cv2.rectangle(output, refPt[0], refPt[1], (0, 255, 0), 2)

refPt = []

img_idx = 0
masks = []
labels = []
rect_list = []
images = []
label_batch_list = []


anno_list = []
bnd_box_list = []

def get_bg_path(idx):
	return bg_dir + str(idx) + ".jpg"


def adjust_gamma(img, gamma=1.0):
	invGamma = 1.0/gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
					  for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(img, table)


def get_save_path():
	global img_idx
	return img_dir + str(img_idx) + ".jpg"


if __name__ == '__main__':

	for filename in os.listdir(img_dir):
		staged_rects = []

		# read image using the absolute path of the file
		img = cv2.imread(img_dir + filename)
		img = cv2.resize(img, (width, height), cv2.INTER_LINEAR)

		img2 = img.copy()                               # a copy of original image
		mask = np.zeros(img.shape[:2],dtype=np.uint8)   # mask initialized to PR_BG
		output = np.zeros(img.shape,np.uint8)           # output image to be shown

		# input and output windows
		cv2.namedWindow('output')
		cv2.namedWindow('input')
		cv2.setMouseCallback('input', onMouseInputImage)
		cv2.setMouseCallback('output', onMouseOutputImage)
		cv2.moveWindow('input', img.shape[1] + 10, 90)

		while(1):
			cv2.imshow('output',output)
			cv2.imshow('input',img)
			k = cv2.waitKey(1)

			# key bindings
			if k == 27:         # esc to exit
				break
			elif k == ord('0'): # BG drawing
				value = DRAW_BG
			elif k == ord('1'): # FG drawing
				value = DRAW_FG
			elif k == ord('2'): # PR_BG drawing
				value = DRAW_PR_BG
			elif k == ord('3'): # PR_FG drawing
				value = DRAW_PR_FG
			elif k == ord('s'): # save image
				if len(rect_list) > 0:
					# the filename of the intial label will not be a real filename
					# instead it will be a seed number for labels created using this label

					# need to generate objects
					# need to label objects

					anno = Annotation(folder=folder, filename=str(img_idx), source="Source", size=Size(width, height, 3))
					print(anno.toPascalVOCFormat())
					anno_list.append(anno)
					masks.append(mask2)
					images.append(img2)
					rect_list = []
					# print("saving label " + str(len(labels)) + " of " + str(num_images))
					# print("saving mask " + str(len(masks)) + " of " + str(num_images))

				img_idx += 1
				print("Result saved as image \n")
				break
			elif k == ord('r'): # reset everything
				print("resetting \n")
				rect = (0,0,1,1)
				anno_list = []
				staged_rects = []
				drawing = False
				rectangle = False
				rect_or_mask = 100
				rect_over = False
				value = DRAW_FG
				img = img2.copy()
				mask = np.zeros(img.shape[:2],dtype = np.uint8) # mask initialized to PR_BG
				output = np.zeros(img.shape,np.uint8)           # output image to be shown
			elif k == ord('w'):
				if refPt is None:
					print("There must be at least one bounding box to proceed.")
				else:
					bndBox = BoundingBox(refPt[0][0], refPt[0][1], refPt[1][0], refPt[1][1])
					print(bndBox.toPascalVOCFormat())
					bnd_box_list.append(bndBox)
			elif k == ord('n'):  # segment the image
				if rect_or_mask == 0:         # grabcut with rect
					bgdmodel = np.zeros((1, 65), np.float64)
					fgdmodel = np.zeros((1, 65), np.float64)
					cv2.grabCut(img2, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
					rect_or_mask = 1
				elif rect_or_mask == 1:         # grabcut with mask
					bgdmodel = np.zeros((1, 65), np.float64)
					fgdmodel = np.zeros((1, 65), np.float64)
					cv2.grabCut(img2, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)

			mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
			output = cv2.bitwise_and(img2, img2, mask=mask2)
			if len(staged_rects) > 0:
				for r in staged_rects:
					if len(r) == 2:
						cv2.rectangle(output, r[0], r[1], (0, 255, 0), 2)

	###########################################################
	# All images have been labeled and masked				  #
	###########################################################

	num_created_samples = 0
	min_scale = 0.45
	max_scale = 1.0
	gen_labels = []
	num_train_sample = 16
	num_test_sample = 0

	print("all images have been labeled and masked.")
	print("creating data...")

	global_counter = 0
	idx = 0
	max_iters_per_fg_and_bg = 25
	# choose a random gamma value between these two values
	min_gamma = .05
	max_gamma = 3.25

	do_test = True

	# number of files in the images directory
	num_images = len([name for name in os.listdir(img_dir) if os.path.isfile(name)])
	# number of files in the background images directory
	num_bg = len([name for name in os.listdir(bg_dir) if os.path.isfile(name)])

	while idx < num_images:
		print("image index: " + str(idx))

		anno = anno_list[idx]
		boxes = bnd_box_list[idx]
		i = images[idx]
		m = masks[idx]
		bg_idx = 0
		for bg_filename in os.listdir(bg_dir):
		# while bg_idx < num_bg:
			print("background index: " + str(bg_idx))
			iters = 0
			# bg_filename = bg_dir + str(bg_idx) + ".jpg"
			# read and resize the background image
			bg_img = cv2.imread(bg_filename)
			bg_img = cv2.resize(bg_img, (width, height), interpolation=cv2.INTER_LINEAR)
			# determine a random gamma adjustment
			rand_bg_gamma = random.uniform(min_gamma, max_gamma)
			# adjust the gamma of the background image
			bg_img = adjust_gamma(bg_img, rand_bg_gamma)
			while iters < max_iters_per_fg_and_bg:
				print("iteration " + str(iters))
				bg_clone = np.copy(bg_img)

				# choose random numbers used to scale the image and mask
				rand_scale_x = random.uniform(min_scale, max_scale)
				rand_scale_y = random.uniform(min_scale, max_scale)

				# scale the mask
				scaled_mask = np.copy(m)
				scaled_mask = cv2.resize(scaled_mask, None, fx=rand_scale_x, fy=rand_scale_y, interpolation=cv2.INTER_LINEAR)
				scaled_mask = cv2.medianBlur(scaled_mask, 7)

				# scale the image
				scaled_image = np.copy(i)
				scaled_image = cv2.resize(scaled_image, None, fx=rand_scale_x, fy=rand_scale_y, interpolation=cv2.INTER_LINEAR)
				rand_fg_gamma = random.uniform(min_gamma, max_gamma)
				scaled_image = cv2.blur(scaled_image, (5, 5))
				scaled_image = adjust_gamma(scaled_image, rand_fg_gamma)

				output = cv2.bitwise_and(scaled_image, scaled_image, mask=scaled_mask)
				print("background image: " + bg_filename)

				h_s, w_s, d_s = scaled_image.shape
				h_bg, w_bg, d_bg = bg_img.shape

				X = int(random.uniform(0, w_bg - w_s - 1))
				Y = int(random.uniform(0, h_bg - h_s - 1))

				processed_bnd_boxes = []
				for b in boxes:
					processed_bnd_boxes.append(BoundingBox(
						int(X + rand_scale_x * b.get('x','min')),
						int(Y + rand_scale_y * b.get('y', 'min')),
						int(X + rand_scale_x * b.get('x', 'max')),
						int(Y + rand_scale_y * b.get('y', 'max'))
					))

				 # processed_rects = []
				 # for r in l.rects:
				 # 	processed_rects.append([[int(X + rand_scale_x * r[0][0]), int(Y + rand_scale_y * r[0][1])],
				 # 							[int(X + rand_scale_x * r[1][0]), int(Y + rand_scale_y * r[1][1])]])

				inverse_mask = cv2.bitwise_not(scaled_mask)
				roi = bg_img[Y:Y+h_s, X:X+w_s]

				masked_roi = cv2.bitwise_and(roi, roi, mask=inverse_mask)
				masked_fg = cv2.bitwise_and(scaled_image, scaled_image, mask=scaled_mask)
				# now remove the black background from the masked foreground\
				# convert image to grayscale
				grayscale_masked_fg = cv2.cvtColor(masked_fg, cv2.COLOR_BGR2GRAY)
				# threshold image to create alpha channel with complete transparency in black background region
				# and zero transparency in foreground object region
				ret, a = cv2.threshold(grayscale_masked_fg, 100, 255, cv2.THRESH_BINARY)
				ret2, a2 = cv2.threshold(a, 100, 255, cv2.THRESH_BINARY)
				# split original image into three single channel images
				b, g, r = cv2.split(masked_fg)
				# merge three single channels and alpha
				merged = cv2.merge((b, g, r, a))
				masked_fg = cv2.GaussianBlur(masked_fg, (21, 21), 0)

				masked_roi = cv2.GaussianBlur(masked_roi, (11, 11), 0)

				combined = cv2.add(masked_fg, masked_roi)
				bg_clone[Y:Y+h_s, X:X+w_s] = combined

				save_path = save_dir + str(idx) + "_" + str(bg_idx) + "_" + str(iters) + ".jpg"
				print("saving to: " + save_path)
				# TODO
				new_anno = Annotation()
				# lab = Label(filename=save_path, rects=processed_rects)
				# gen_labels.append(lab)
				print("num labels: " + str(len(gen_labels)))
				# print("generated label: " + str(lab))
				cv2.imwrite(save_path, bg_clone)

				num_created_samples += 1
				print("created sample " + str(num_created_samples) + " of " + str(num_test_sample + num_train_sample))
				iters += 1
			print("incrementing bg index " + str(bg_idx))
			bg_idx += 1

		print("shuffling labels")
		shuffle(gen_labels)  # single hand over multiple backgrounds
		shuffle(label_batch_list)
		label_batch_list.append(gen_labels)
		for batch in label_batch_list:
			shuffle(batch)

		print("saving labels")
		print("created train samples " + str(num_created_samples) + " of " + str(num_train_sample))
		# print(process_labels(gen_labels))

		# append to the label output file
		train_label_file = "D:/work/training-data/generated/labels/train_boxes.json"
		train_json = open(train_label_file, "w+")
		# train_json.write(process_labels(gen_labels))
		train_json.write(",")
		train_json.close()

		print("incrementing image index " + str(idx))
		idx += 1

	cv2.destroyAllWindows()
