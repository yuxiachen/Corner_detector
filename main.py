import matplotlib.image as mpimg
import numpy as np
import argparse
from harris_detector import harris_detector
from sift_detector import sift_detector
import cv2


def draw_marker(img, filename, position_1=None, position_2=None):
	temp = np.copy(img)
	t_delta = 3
	if position_1 is not None:
		for i in range(0, len(position_1)):
			pts = np.array([[position_1[i][0], position_1[i][1] - t_delta], 
				[position_1[i][0] - t_delta,position_1[i][1] + t_delta], 
				[position_1[i][0] + t_delta,position_1[i][1] + t_delta]],
				np.int32)
			pts = pts.reshape((-1,1,2))
			temp = cv2.polylines(
				temp, 
				[pts], 
				True, 
				color=(1,0,0,1), 
				thickness=1
			)

	if position_2 is not None:
		for i in range(0, len(position_2)):

			temp = cv2.rectangle(
				temp,
    			(position_2[i][0] + t_delta, position_2[i][1] + t_delta), 
    			(position_2[i][0] - t_delta, position_2[i][1] - t_delta),
    			color=(0,1,0,1),
    			thickness=1
			)
			
	mpimg.imsave(filename, temp)


# Main function
def main(args):
	image_file = args.image_file
	output_file_1 = args.output_file_1
	output_file_2 = args.output_file_2
	output_file_3 = args.output_file_3
	threshold_1 = args.threshold_1
	threshold_2 = args.threshold_2

	img = Image.open(image_file)

	print('processing image...')

	ret_1 = harris_detector(img[:, :, 0] * 255, threshold_1)
	ret_2 = sift_detector(img[:, :, 0] * 255, threshold_2)

	print('drawing marker on the detected points...')

	draw_marker(img, output_file_1, position_1 = ret_1)
	draw_marker(img, output_file_2, position_2 = ret_2)

	draw_marker(img, output_file_3, position_1 = ret_1, position_2 = ret_2)


# Define the args to pass in the input files, output files and thresholds
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--image_file', type=str, 
		help='source image file to process')
	parser.add_argument('--output_file_1', type=str, 
		help='image file to store the harris result.')
	parser.add_argument('--output_file_2', type=str, 
		help='image file to store the sift result.')
	parser.add_argument('--output_file_3', type=str, 
		help='image file to store the overlay result.')
	parser.add_argument('--threshold_1', type=int, default=None, 
		help='the threshold used to binarize the image')
	parser.add_argument('--threshold_2', type=int, default=None, 
		help='the threshold used to binarize the image')

	args = parser.parse_args()
	main(args)


