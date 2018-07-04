import glob as glob
import numpy as np
import cv2
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import tensorflow as tf

# path = "/media/mnt/dehaze/aod_logs/aod_vanilla_newData2nchance/results/val/"
path = "/media/mnt/dehaze/data/cityscape/processed/pix2pix-tensorflow/logs/1st_train/result/val/images/"

images = glob.glob(path+"*-outputs.png")
print len(images)

psnr_list = []
ssim_list = []

for image in images:
	img1 = cv2.imread(image)
	img2 = cv2.imread(image[:-11]+"targets.png")
	# img1  = img[:,320:640,:]
	# img2 = img[:,640:,:]
	psnr_val = psnr(img1, img2)
	ssim_val = ssim(img1, img2, gaussian_weights=True, multichannel=True)
	psnr_list.append(psnr_val)
	ssim_list.append(ssim_val)
	print psnr_val, ssim_val	

av_psnr = np.mean(psnr_list)
av_ssim = np.mean(ssim_list)

print "Average PSNR:", av_psnr 
print "Average SSIM:", av_ssim

with open(path+"metrix.txt", "w") as file:
	file.write("Average PSNR Value: %.2f \n" % av_psnr)
	file.write("Average SSIM Value: %.4f \n" % av_ssim)

	



# img1_in = tf.placeholder(tf.float32, shape=[240,320,3], name="Haze_Image")
# img2_in = tf.placeholder(tf.float32, shape=[240,320,3], name="Clear_Image")

# ssim_op = tf.image.ssim(img1_in, img2_in, max_val=255)
# psnr_op = tf.image.psnr(img1_in, img2_in, max_val=255)

# sess = tf.Session()

# out = sess.run([ssim_op, psnr_op],{img1_in:img1, img2_in:img2})

# print "tf psnr:", out[1], "tf ssim:", out[0]
