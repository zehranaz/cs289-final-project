#!/usr/bin/env python
# Convert list of coordinates into pictures.
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import wiener
import math
import cv2

def enhance_image(input_image):
    my_im = input_image.copy()
    my_im = wiener(my_im)
    return my_im

def make_image_from_coords(coordinates, name):
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

	# thinning
	img = cv2.imread('lao_images/' + name + '.bmp')
	'''
	cv2.imshow('Original',img)
	cv2.waitKey(0)

	gray_im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cv2.imshow('Gray',gray_im)
	cv2.waitKey(0)

	enhanced_im = enhance_image(gray_im)
	enhanced_im  = np.array(enhanced_im, np.uint8)
	ret, thresh1 = cv2.threshold(enhanced_im,250,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	cv2.imshow('thresh1', thresh1)
	cv2.waitKey(0)

	
	plt.axis("off")
	plt.imshow(thresh1)
	plt.show()
	

	kernel = np.ones((2,2),np.uint8)
	erosion = cv2.erode(thresh1, kernel, iterations = 1)
	cv2.imshow('eroded', erosion)

	cv2.imwrite('lao_images/eroded' + name + '.bmp', erosion)
	
	plt.axis("off")
	plt.imshow(erosion)
	plt.show()
	'''

	# return width and height of image
	return (max_x - min_x, max_y - min_y)

def print_pixels_in_image (image, max_x, max_y):
	for i in range(max_y + 1):
		for j in range(max_x + 1):
			print str(image[i][j]) + " "
		print "\n"

# extract coordinates for characters in char_lst
def extract_coordinates_from_file(filename, char_lst):
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
	char_index = 0
	for line in f:
		line_num += 1
		# skip first ten lines
		if line_num < 10:
			continue
		tokens = line.split(" ")
		if len(tokens) == 3 and char_index in char_lst:
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
			if len(coordinates) > 0 and char_index in char_lst:
				coordinates_list.append(coordinates)
				coordinates = []
			char_index += 1
			# print tokens

	# last character's coordinates			
	if char_index in char_lst and not len(coordinates) == 0:
		coordinates_list.append(coordinates)
	return coordinates_list

def make_images_from_file(filename, char_lst):
	file_token = filename.split('/')[-1].split('.')[0]
	coordinate_list = extract_coordinates_from_file(filename, char_lst)
	item = 0
	img_sizes = []
	for i, lst in enumerate(coordinate_list):
		img_sizes.append(make_image_from_coords(lst, file_token + "_" + str(char_lst[i])))
	return img_sizes

def convert(fname, person_lst, char_lst): 
	# test case
	# print extract_coordinates_from_file("Distribution/Tests/interpolate_test.txt")
	img_sizes = []
	for pind in person_lst:
	 	img_sizes += make_images_from_file(fname + "000" + str(pind) + ".txt", char_lst)
	return img_sizes
	"""
	for i in range(10, 52):
		make_images_from_file("Distribution/Data/00" + str(i) + ".txt")
	"""

# scales a picture to fit a fixed height and width, maintaining aspect ratio
def fit_to_size(fname, basewidth, baseheight):
	# Rescale so width = sizex (keep aspect ratio)
	old_im = Image.open(fname) 
	wpercent = (basewidth/float(old_im.size[0])) 
	# we want the highest predicted hsize
	hsize = int((float(old_im.size[1])*float(wpercent))) 
	new_im = old_im.resize((basewidth,hsize), PIL.Image.ANTIALIAS) 

	# Add margins so image is at fixed height and width
	old_im = new_im
	old_size = old_im.size 
	new_size = (basewidth, baseheight) # new image size is exactly 800 x 800, centered
	new_im = Image.new("RGB", new_size) ## luckily, this is already black! 
	new_im.paste(old_im, ((new_size[0]-old_size[0])/2, (new_size[1]-old_size[1])/2))
	new_im.save(fname)

def convert_images(person_lst, char_lst):
	img_sizes = convert("Distribution/Data/", person_lst, char_lst)

	# determine the width and height each image should be scaled to
	x, y = [i[0] for i in img_sizes], [i[1] for i in img_sizes]
	basewidth = max(x) # minimum basewidth
	multipliers_lst = map(lambda x: basewidth/float(x), x)
	height_lst = [mult*height for mult, height in zip(multipliers_lst, y)]
	baseheight = int(math.ceil(max(height_lst)))

	# scale each image
	for person_index in person_lst:
		for char_index in char_lst:
			filename = "lao_images/000" + str(person_index) + "_" + str(char_index) + ".bmp"
			fit_to_size(filename, basewidth, baseheight)
'''
def main():
	""" for Tamil characters dataset
    # convert tiff file to binary array
    im = plt.imread(tiff_file) # converts to type numpy.ndarray
    nrow = im.shape[0]
    ncol = im.shape[1]
    binim = np.zeros((nrow, ncol)) # 2D binary array
    for row in range(nrow):
        for col in range(ncol):
            # 1 if black, 0 if white
            binim[row][col] = 1 if 0 in im[row][col] else 0
    """
	# pick the people and character indices that you want to convert
	person_lst = [1, 2] # works for persons 0-9
	char_lst = [1,11,12]
	img_sizes = convert("Distribution/Data/", person_lst, char_lst)

	# determine the width and height each image should be scaled to
	x, y = [i[0] for i in img_sizes], [i[1] for i in img_sizes]
	basewidth = max(x) # minimum basewidth
	multipliers_lst = map(lambda x: basewidth/float(x), x)
	height_lst = [mult*height for mult, height in zip(multipliers_lst, y)]
	baseheight = int(math.ceil(max(height_lst)))

	# scale each image
	for person_index in person_lst:
		for char_index in char_lst:
			filename = "lao_images/000" + str(person_index) + "_" + str(char_index) + ".bmp"
			fit_to_size(filename, basewidth, baseheight)

if __name__ == "__main__":
    main()
'''