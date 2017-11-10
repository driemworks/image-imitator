import os


class Directories:
	# generate this many training images
	num_generated_images = 12000

	# size of images
	# input and output images will be resized to be these sizes
	image_width = 640
	image_height = 480
	image_depth = 3

	segmented = 0

	num_iters_per_fg_and_bg = 1

	def __init__(self):
		self.root_dir = os.path.abspath(os.path.dirname(__file__))
		# image directories
		self.input_image_directory = self.root_dir + "/images/samples/input/"
		self.background_image_directory = self.root_dir + "/images/samples/background/"

		# save dirs
		self.save_image_dir = self.root_dir + "/images/generated/images/"
		self.save_anno_dir = self.root_dir + "/images/generated/anno/"







# labels for extracted objects
class Labels:
	thumb = "thumb"
	index = "index"
	middle = "middle"
	ring = "ring"
	pinky = "pinky"

	def get_labels(self):
		return [self.thumb, self.index, self.middle, self.ring, self.pinky]
