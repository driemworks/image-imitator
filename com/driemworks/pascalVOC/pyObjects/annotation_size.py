class Size:

	def __init__(self, width, height, depth):
		self.width = width
		self.height = height
		self.depth = depth

	def toPascalVOCFormat(self):
		output = "<size>"
		output += "\n\t<width>" + str(self.width) + "</width>"
		output += "\n\t<height>" + str(self.height) + "</height>"
		output += "\n\t<depth>" + str(self.depth) + "</depth>"
		output += "\n</size>"
		return output