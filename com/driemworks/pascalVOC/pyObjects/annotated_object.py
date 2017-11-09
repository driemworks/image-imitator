from com.driemworks.pascalVOC.formatter.format_utils import appendPascalVOC
from com.driemworks.pascalVOC.pyObjects.BndBox import BoundingBox


class Annotated_Object:

	def __init__(self, name, pose, truncated, difficult, bndBox):
		self.name = name
		self.pose = pose
		self.truncated = truncated
		self.difficult = difficult
		self.bndBox = bndBox

	def toPascalVOCFormat(self):
		output = "<object>"
		output += "\n\t<name>" + self.name + "</name>"
		output += "\n\t<pose>" + self.pose + "</pose>"
		output += "\n\t<truncated>" + self.truncated + "</truncated>"
		for box in self.bndBox:
			output += appendPascalVOC(output, box.toPascalVOCFormat())
			# output += "\n\t" + box.toPascalVOCFormat()
		output += "\n</object>"
		return output

# boxes = [BoundingBox(0,0,0,0)]
# anno_obj = Annotated_Object("name", "pose", "truncated", "difficult", boxes)
# print("=========================")
# print(anno_obj.toPascalVOCFormat())