# Converty array of coordinates into pictures.
import numpy as np
from PIL import Image

#coordinates = [(1, 0), (2, 1), (3, 2), (4,3), (1,3), (0,3), (3,4), (2,4)]

def make_image_from_coords (coordinates, name):
	x, y = [i[0] for i in coordinates], [i[1] for i in coordinates]
	max_x, max_y = max(x), max(y)
	min_x, min_y = min(x), min(y)

	image = np.zeros((max_y - min_y + 1, max_x - min_x + 1))

	for i in range(len(coordinates)):
	    image[y[i] - min_y][x[i] - min_x] = int(1)

	im = Image.fromarray(image.astype('uint8')*255)

	im.save('lao_images/' + name + '.bmp')

def print_pixels_in_image (image, max_x, max_y):
	for i in range(max_y + 1):
		for j in range(max_x + 1):
			print str(image[i][j]) + " "
		print "\n"

def extract_coordinates_from_file(filename):
	f = open(filename, 'r')
	line_num = 0
	coordinates_list = []
	coordinates = []
	for line in f:
		line_num += 1
		if line_num < 10:
			continue
		tokens = line.split(" ")
		if len(tokens) == 3:
			coordinates.append((int(tokens[1]), int(tokens[2])))
		if len(tokens) == 1:
			if not len(coordinates) == 0:
				coordinates_list.append(coordinates)
				coordinates = []
	if not len(coordinates) == 0:
		coordinates_list.append(coordinates)
	return coordinates_list

def make_images_from_file(filename):
	file_token = filename.split('/')[6].split('.')[0]
	coordinate_list = extract_coordinates_from_file(filename)
	item = 0
	for lst in coordinate_list:
		make_image_from_coords(lst, file_token + "_" + str(item))
		item += 1

def convert (): 
	for i in range(1, 9):
	 	make_images_from_file("/home/zehranaz/genetic/Distribution/Data/000" + str(i) + ".txt")
	for i in range(10, 52):
		make_images_from_file("/home/zehranaz/genetic/Distribution/Data/00" + str(i) + ".txt")

#convert()