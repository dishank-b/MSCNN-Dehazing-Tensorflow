import numpy as np
import cv2
import glob

x_img_list=glob.glob("/media/mnt/dehaze/*.jpg")

final_width = 320	
final_height = 240	
	
for image in x_img_list:
	# print image		
	img = cv2.imread(image)
	w = img.shape[1]
	h = img.shape[0]
	ar = float(w)/float(h)
	if w/final_width<h/final_height:
		new_w = final_width
		new_h = int(new_w/ar)
		a = new_h - final_height
		resize_img = cv2.resize(img, dsize=(new_w, new_h))
		final_image = resize_img[a/2:a/2+final_height,:]
	elif w/final_width>h/final_height:
		new_h = final_height
		new_w = int(new_h*ar)
		a = new_w - final_width
		resize_img = cv2.resize(img,dsize=(new_w, new_h))
		final_image = resize_img[:,a/2:a/2+final_width]
	else:
		resize_img = cv2.resize(img,dsize=(final_width, final_height))
		final_image = resize_img

	# final_image = final_image[:216, :]
	cv2.imwrite(image[:-4]+"_resize.jpg", final_image)