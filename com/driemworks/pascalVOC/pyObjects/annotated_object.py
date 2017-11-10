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
		output += "\n\t<name>" + str(self.name) + "</name>"
		output += "\n\t<pose>" + str(self.pose) + "</pose>"
		output += "\n\t<truncated>" + str(self.truncated) + "</truncated>"
		output = appendPascalVOC(output, self.bndBox.toPascalVOCFormat())
		output += "\n</object>"
		return output

# boxes = [BoundingBox(0,0,0,0)]
# anno_obj = Annotated_Object("name", "pose", "truncated", "difficult", boxes)
# print("=========================")
# print(anno_obj.toPascalVOCFormat())