from com.driemworks.pascalVOC.formatter.format_utils import appendPascalVOC
from com.driemworks.pascalVOC.pyObjects.BndBox import BoundingBox
from com.driemworks.pascalVOC.pyObjects.annotated_object import Annotated_Object
from com.driemworks.pascalVOC.pyObjects.annotation_size import Size


class Annotation:

	def __init__(self, folder, filename, source, size, segmented, objects):
		self.folder = folder
		self.filename = filename
		self.source = source
		self.size = size
		self.segmented = segmented
		self.objects = objects

	def toPascalVOCFormat(self):
		output = "<annotation>"
		output += "\n\t<folder>" + self.folder + "</folder>"
		output += "\n\t<filename>" + self.filename + "</filename>"
		output += "\n\t<source>"
		output += "\n\t\t<database>" + self.source + "</database>"
		output += "\n\t<\source>"
		output = appendPascalVOC(output, self.size.toPascalVOCFormat())
		for object in self.objects:
			output = appendPascalVOC(output, object.toPascalVOCFormat())
		output += "\n</annotation>"
		return output

# folder = "folder"
# filename = "filename"
# source = "Source"
# size = Size(str(400), str(400), "Unspecified")
# segmented = "Unspecified"
# boxes = [BoundingBox(0,0,1,1)]
# anno_objs = [Annotated_Object("name", "pose", "truncated", "difficult", boxes)]
#
# print("===================================")
# anno = Annotation(folder,filename, source, size, segmented, anno_objs)
# print(anno.toPascalVOCFormat())

