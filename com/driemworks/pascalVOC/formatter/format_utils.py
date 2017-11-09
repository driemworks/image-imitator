def appendPascalVOC(head, tail):
	output = head
	if "\n" in output:
		lines = tail.split("\n")
		for line in lines:
			output += "\n\t" + line
	return output
