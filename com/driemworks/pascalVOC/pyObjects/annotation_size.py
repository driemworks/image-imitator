from config import Directories as config


class Size:

	def __init__(self, width, height, depth):
		self.width = width
		self.height = height
		self.depth = depth

	# the default size is the size specified in config.py
	@staticmethod
	def get_default_size():
		return Size(config.image_width, config.image_height, config.image_depth)

	def toPascalVOCFormat(self):
		output = "<size>"
		output += "\n\t<width>" + str(self.width) + "</width>"
		output += "\n\t<height>" + str(self.height) + "</height>"
		output += "\n\t<depth>" + str(self.depth) + "</depth>"
		output += "\n</size>"
		return output