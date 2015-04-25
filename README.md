# cs289-final-project
Off-line handwriting recognition project for CS 289.

Image processing
* originally did image resizing and padding, which resulted in grayscale image that was no longer unit length
* added code that implemented Zhang Suen algorithm for thinning to unit length https://github.com/bsdnoobz/zhang-suen-thinning/blob/master/thinning.py
* after that, it was much easier to recursively iterate through the array and create a graph
