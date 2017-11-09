class BoundingBox:

	def __init__(self, x_min, y_min, x_max, y_max):
		self.x_min = x_min
		self.y_min = y_min
		self.x_max = x_max
		self.y_max = y_max

	def get(self, axis, extrema):
		if str(axis) == 'x':
			if str(extrema) == 'min':
				return self.x_min
			else:
				return self.x_max
		elif str(axis) == 'y':
			if str(extrema) == 'min':
				return self.y_min
			else:
				return self.y_max

	# get the bounding box in the pascal voc format
	def toPascalVOCFormat(self):
		output = "<bndBox>"
		output += "\n\t<xmin>" + str(self.x_min) + "</xmin>"
		output += "\n\t<ymin>" + str(self.y_min) + "</ymin>"
		output += "\n\t<xmax>" + str(self.x_max) + "</xmax>"
		output += "\n\t<ymax>" + str(self.y_max) + "</ymax>"
		output += "\n</bndBox>"
		return output


# bndBox = BoundingBox(x_min=0, y_min=0, x_max=0, y_max=0)
# print(bndBox.toPascalVOCFormat())