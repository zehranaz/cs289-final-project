# Converty array of coordinates into pictures.
import numpy as np
from PIL import Image

# TO DO: Center Images

#coordinates = [(1, 0), (2, 1), (3, 2), (4,3), (1,3), (0,3), (3,4), (2,4)]

def make_image_from_coords (coordinates, name):
	x, y = [i[0] for i in coordinates], [i[1] for i in coordinates]
	max_x, max_y = max(x), max(y)
	min_x, min_y = min(x), min(y)

	image = np.zeros((max_y - min_y + 1, max_x - min_x + 1))

	for i in range(len(coordinates)):
	    image[y[i] - min_y][x[i] - min_x] = int(1)

	im = Image.fromarray(image.astype('uint8')*255)

	# save as binary array text file
	np.savetxt('lao_text/' + name + '.txt', image, fmt='%d')

	# save as bitmap image
	im.save('lao_images/' + name + '.bmp')

def print_pixels_in_image (image, max_x, max_y):
	for i in range(max_y + 1):
		for j in range(max_x + 1):
			print str(image[i][j]) + " "
		print "\n"

def extract_coordinates_from_file(filename):
	f = open(filename, 'r')
	line_num = 0
	# for each person
	coordinates_list = []
	# one for each character
	coordinates = []
	# interpolate in cases where x-coords or 
	#	y-coords differ from prev by this threshold
	dist_to_interpolate = 2
	prev_x = None
	prev_y = None
	for line in f:
		line_num += 1
		# skip first ten lines
		if line_num < 10:
			continue
		tokens = line.split(" ")
		if len(tokens) == 3:
			new_x = int(tokens[1])
			new_y = int(tokens[2])
			if prev_x and prev_y:
				# interpolate if separated from prev x or y coord
				if abs(new_x - prev_x) == dist_to_interpolate or \
					abs(new_y - prev_y) == dist_to_interpolate:
					mid_x = prev_x + int(round((new_x - prev_x)/2))
					mid_y = prev_y + int(round((new_y - prev_y)/2))
					coordinates.append((mid_x, mid_y))
			coordinates.append((new_x, new_y))
			prev_x, prev_y = new_x, new_y

		# reached end of character line
		if len(tokens) == 1:
			if len(coordinates) > 0:
				coordinates_list.append(coordinates)
				coordinates = []

	# last character's coordinates			
	if not len(coordinates) == 0:
		coordinates_list.append(coordinates)
	return coordinates_list

def make_images_from_file(filename):
	file_token = filename.split('/')[-1].split('.')[0]
	coordinate_list = extract_coordinates_from_file(filename)
	item = 0
	for lst in coordinate_list:
		make_image_from_coords(lst, file_token + "_" + str(item))
		item += 1

def convert (): 
	# test case
	# print extract_coordinates_from_file("Distribution/Tests/interpolate_test.txt")
	for i in range(1, 2):
	 	make_images_from_file("Distribution/Data/000" + str(i) + ".txt")
	"""
	for i in range(10, 52):
		make_images_from_file("Distribution/Data/00" + str(i) + ".txt")
	"""

convert()
