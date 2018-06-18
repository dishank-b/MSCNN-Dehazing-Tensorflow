# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import tensorflow as tf
import numpy as np
import cv2
from ops import *
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils import *

class MSCNN(object):
	"""Single Image Dehazing via Multi-Scale Convolutional Neural Networks"""
	def __init__(self, model_path):
		self.model_path = model_path
		self.graph_path = model_path+"/tf_graph/"
		self.save_path = model_path + "/saved_model/"
		self.output_path = model_path + "/results/"
		if not os.path.exists(model_path):
			os.makedirs(self.graph_path+"train/")
			os.makedirs(self.graph_path+"val/")
		
	def _debug_info(self):
		variables_names = [[v.name, v.get_shape().as_list()] for v in tf.trainable_variables()]
		print "Trainable Variables:"
		tot_params = 0
		for i in variables_names:
			var_params = np.prod(np.array(i[1]))
			tot_params += var_params
			print i[0], i[1], var_params
		print "Total number of Trainable Parameters: ", str(tot_params/1000.0)+"K"

	def _coarseNet(self, x):
		with tf.variable_scope("coarseNet") as var_scope:
			# conv1 = Conv_2D(x, output_chan=5, kernel=[11,11], stride=[1,1], padding="SAME", name="Conv1")
			conv1 = Conv_2D(x, output_chan=5, kernel=[11,11], stride=[2,2], padding="SAME", train_phase=self.train_phase,name="Conv1")
			# pool1 = max_pool(conv1, 2, 2, "max_pool1")
			# upsample1 = max_unpool(pool1, "upsample1")
			upsample1 = Dconv_2D(conv1, output_chan=5, kernel=[5,5], stride=[2,2], padding="SAME", train_phase=self.train_phase,name="D_Conv1")

			# conv2 = Conv_2D(upsample1, output_chan=5, kernel=[9,9], stride=[1,1], padding="SAME", name="Conv2")
			conv2 = Conv_2D(upsample1, output_chan=5, kernel=[9,9], stride=[2,2], padding="SAME", train_phase=self.train_phase,name="Conv2")
			# pool2 = max_pool(conv2, 2, 2, "max_pool2")
			# upsample2 = max_unpool(pool2, "upsample2")
			upsample2 = Dconv_2D(conv2, output_chan=5, kernel=[5,5], stride=[2,2], padding="SAME", train_phase=self.train_phase,name="D_Conv2")
	
			# conv3 = Conv_2D(upsample2, output_chan=10, kernel=[7,7], stride=[1,1], padding="SAME", name="Conv3")
			conv3 = Conv_2D(upsample2, output_chan=10, kernel=[7,7], stride=[2,2], padding="SAME", train_phase=self.train_phase,name="Conv3")
			# pool3 = max_pool(conv3, 2, 2, "max_pool3")
			# upsample3 = max_unpool(pool3, "upsample3")	
			upsample3 = Dconv_2D(conv3, output_chan=10, kernel=[5,5], stride=[2,2], padding="SAME", train_phase=self.train_phase,name="D_Conv3")	

			linear = Conv_2D(upsample3, output_chan=1, kernel=[1,1], stride=[1,1], padding="SAME", activation=tf.sigmoid,train_phase=self.train_phase, 
							add_summary=True, name="linear_comb")
			return linear

	def _fineNet(self, x, coarseMap):
		with tf.variable_scope("fineNet") as var_scope:
			# conv1 = Conv_2D(x, output_chan=4, kernel=[7,7], stride=[1,1], padding="SAME", name="Conv1")
			conv1 = Conv_2D(x, output_chan=4, kernel=[7,7], stride=[2,2], padding="SAME", train_phase=self.train_phase, name="Conv1")
			# pool1 = max_pool(conv1, 2, 2, "max_pool1")
			# upsample1 = max_unpool(pool1, "upsample1")
			upsample1 = Dconv_2D(conv1, output_chan=4, kernel=[5,5], stride=[2,2], padding="SAME", train_phase=self.train_phase, name="D_Conv1")
			
			concat = tf.concat([upsample1, coarseMap], axis=3, name="CorMapConcat")

			# conv2 = Conv_2D(concat, output_chan=5, kernel=[5,5], stride=[1,1], padding="SAME", name="Conv2")
			conv2 = Conv_2D(concat, output_chan=5, kernel=[5,5], stride=[2,2], padding="SAME", train_phase=self.train_phase,name="Conv2")
			# pool2 = max_pool(conv2, 2, 2, "max_pool2")
			# upsample2 = max_unpool(pool2, "upsample2")
			upsample2 = Dconv_2D(conv2, output_chan=5, kernel=[5,5], stride=[2,2], padding="SAME", train_phase=self.train_phase, name="D_Conv2")
	
			# conv3 = Conv_2D(upsample2, output_chan=10, kernel=[3,3], stride=[1,1], padding="SAME", name="Conv3")
			conv3 = Conv_2D(upsample2, output_chan=10, kernel=[3,3], stride=[2,2], padding="SAME", train_phase=self.train_phase, name="Conv3")
			# pool3 = max_pool(conv3, 2, 2, "max_pool3")
			# upsample3 = max_unpool(pool3, "upsample3")	
			upsample3 = Dconv_2D(conv3, output_chan=10, kernel=[5,5], stride=[2,2], padding="SAME", train_phase=self.train_phase, name="D_Conv3")	

			linear = Conv_2D(upsample3, output_chan=1, kernel=[1,1], stride=[1,1], padding="SAME", activation=tf.sigmoid,train_phase=self.train_phase,
							add_summary=True, name="linear_comb")
			
			return linear

	def build_model(self):
		with tf.name_scope("Inputs") as scope:
			self.haze_in = tf.placeholder(tf.float32, shape=[None,240,320,3], name="Haze_Image")
			self.clear_in = tf.placeholder(tf.float32, shape=[None,240,320,3], name="Clear_Image")
			self.trans_in = tf.placeholder(tf.float32, shape=[None,240,320,1], name="TMap")
			self.train_phase = tf.placeholder(tf.bool, name="is_training")
			hazy_summ = tf.summary.image("Hazy_image", self.haze_in)
			map_summ = tf.summary.image("Trans_Map", self.trans_in)
			clear_summ = tf.summary.image("clear_in", self.clear_in)

		with tf.name_scope("Model") as scope:
			self.coarseMap = self._coarseNet(self.haze_in)
			self.transMap = self._fineNet(self.haze_in, self.coarseMap)
			self.clearImg = getClearImage(self.haze_in, self.transMap)
			# self.clearImg = getClearImage(self.haze_in, self.trans_in)
			clear_image_summ = tf.summary.image("Out_Clear", self.clearImg)

		with tf.name_scope("Loss") as scope:
			self.coarseLoss = tf.losses.mean_squared_error(self.clear_in, self.clearImg)\
							+ tf.losses.mean_squared_error(self.trans_in, self.coarseMap)
			self.fineLoss = tf.losses.mean_squared_error(self.clear_in, self.clearImg)\
						  + tf.losses.mean_squared_error(self.trans_in, self.transMap)
							 
			self.coarse_loss_summ = tf.summary.scalar("Coarse Loss", self.coarseLoss)
			self.fine_loss_summ = tf.summary.scalar("Fine Loss", self.fineLoss)

		with tf.name_scope("Optimizers") as scope:
			train_vars = tf.trainable_variables()
			self.coarse_vars = [var for var in train_vars if "coarse" in var.name]
			self.fine_vars = [var for var in train_vars if "fine" in var.name]

			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  			with tf.control_dependencies(update_ops):
				self.coarse_solver = tf.train.AdamOptimizer(learning_rate=1e-04).minimize(self.coarseLoss, var_list=self.coarse_vars)
				self.fine_solver = tf.train.AdamOptimizer(learning_rate=1e-04).minimize(self.fineLoss, var_list=self.fine_vars)

		self.merged_summ = tf.summary.merge_all()
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=config)
		self.train_writer = tf.summary.FileWriter(self.graph_path+"train/")
		self.train_writer.add_graph(self.sess.graph)
		self.val_writer = tf.summary.FileWriter(self.graph_path+"val/")
		self.val_writer.add_graph(self.sess.graph)
		self.saver = tf.train.Saver()
		self.sess.run(tf.global_variables_initializer())
		self._debug_info()

	def train_model(self, train_imgs, val_imgs, learning_rate=1e-5, batch_size=32, epoch_size=50):
		
		print "Training Images: ", train_imgs[0].shape[0]
		print "Validation Images: ", val_imgs[0].shape[0]
		print "Learning_rate: ", learning_rate, "Batch_size", batch_size, "Epochs", epoch_size
		raw_input("Training will start above configuration. Press Enter to Start....")
		
		with tf.name_scope("Training") as scope:
			for epoch in range(epoch_size):
				for itr in xrange(0, train_imgs[0].shape[0]-batch_size, batch_size):
					haze_in = train_imgs[0][itr:itr+batch_size][:,0,:,:,:]
					clear_in = train_imgs[0][itr:itr+batch_size][:,1,:,:,:]
					trans_in = train_imgs[1][itr:itr+batch_size]

					sess_in = [self.coarse_solver, self.coarseLoss, self.merged_summ]
					coarse_out = self.sess.run(sess_in, {self.haze_in:haze_in, self.trans_in:trans_in, self.clear_in: clear_in,self.train_phase:True})
					self.train_writer.add_summary(coarse_out[2])
					
					sess_in  = [self.fine_solver, self.fineLoss, self.merged_summ]
					fine_out = self.sess.run(sess_in, {self.haze_in:haze_in, self.trans_in:trans_in, self.clear_in: clear_in,self.train_phase:True})
					self.train_writer.add_summary(fine_out[2])

					if itr%5==0:
						print "Epoch: ", epoch, "Iteration: ", itr, " Coarse Loss: ", coarse_out[1], "Fine Loss: ", fine_out[1], \
								"Loss: ", coarse_out[1]+fine_out[1]


				for itr in xrange(0, val_imgs[0].shape[0]-batch_size, batch_size):
					haze_in = val_imgs[0][itr:itr+batch_size][:,0,:,:,:]
					clear_in = val_imgs[0][itr:itr+batch_size][:,1,:,:,:]
					trans_in = val_imgs[1][itr:itr+batch_size]

					c_val_loss,f_val_loss ,summ = self.sess.run([self.coarseLoss,self.fineLoss ,self.merged_summ], {self.haze_in: haze_in, 
												self.trans_in: trans_in,self.clear_in: clear_in, self.train_phase:False})
					# c_val_loss, summ = self.sess.run([self.coarseLoss, self.merged_summ], {self.haze_in: haze_in, 
					# 							self.trans_in: trans_in,self.clear_in: clear_in, self.train_phase:False})
					self.val_writer.add_summary(summ)

					print "Epoch: ", epoch, "Iteration: ", itr, "Validation Loss: ", c_val_loss+f_val_loss

				if epoch%10==0:
					self.saver.save(self.sess, self.save_path+"MSCNN", global_step=epoch)
					print "Checkpoint saved"

					a = np.random.randint(1, train_imgs[0].shape[0], 1)
					
					random_img = train_imgs[0][a]

					gen_imgs = self.sess.run(self.clearImg, {self.haze_in: random_img[:,0,:,:,:], self.trans_in: train_imgs[1][a], self.train_phase:False})
					for i,j,k in zip(random_img[:,0,:,:,:], random_img[:,1,:,:,:], gen_imgs):
						stack = np.hstack((i,j,k))
						cv2.imwrite(self.output_path +str(epoch)+"_train_img.jpg", 255.0*stack)

	# def test(self, input_imgs, batch_size):
	def test(self, batch_size):
		sess=tf.Session()
		
		saver = tf.train.import_meta_graph(self.save_path+'MSCNN-90.meta')
		print self.save_path
		saver.restore(sess,tf.train.latest_checkpoint(self.save_path))
		print self.save_path
		graph = tf.get_default_graph()

		x = graph.get_tensor_by_name("Inputs/Haze_Image:0")
		is_train = graph.get_tensor_by_name("Inputs/is_training:0")
		y = graph.get_tensor_by_name("Model/fineNet/linear_comb/Sigmoid:0")
		# y = graph.get_tensor_by_name("Model/fineNet/linear_comb/Relu:0")
		clr_img_clip = graph.get_tensor_by_name("Model/clearImage/clip_by_value:0")
		clr_img = graph.get_tensor_by_name("Model/clearImage/concat_1:0") 
		airlight = graph.get_tensor_by_name("Model/clearImage/concat:0")
		
		# print "Tensor Loaded"
		# for itr in xrange(0, input_imgs.shape[0], batch_size):
		# 	if itr+batch_size<=input_imgs.shape[0]:
		# 		end = itr+batch_size
		# 	else:
		# 		end = input_imgs.shape[0]
		# 	input_img = input_imgs[itr:end]
		# 	out = sess.run([y, clr_img_clip, airlight], {x:input_img, is_train:False})
		# 	if itr==0:
		# 		tot_maps = out[0]
		# 		tot_clr = out[1]
		# 		tot_air = out[2]
		# 	else:
		# 		tot_maps = np.concatenate((tot_maps, out[0]))
		# 		tot_clr = np.concatenate((tot_clr, out[1]))
		# 		tot_air = np.concatenate((tot_airn, out[2]))
		# print "Output Shape: ", tot_maps.shape, tot_clr.shape, tot_air.shape
		# return tot_maps, tot_clr, tot_air

		img_name = glob.glob("/media/mnt/dehaze/*_resize.jpg")
		for image in img_name:
			img = cv2.imread(image)
			in_img = img.reshape((1, img.shape[0],img.shape[1],img.shape[2]))
			out = sess.run([y, clr_img_clip, clr_img, airlight], {x:in_img/255.0, is_train:False})
			# plt.imshow(out[2][0])
			# plt.show()
			# plt.imshow(out[3][0])
			# plt.show()
			# maps = sess.run(y, {x:in_img/255.0, is_train:False})
			# clear = clearImg(img, maps[0])
			cv2.imwrite(image[:-4]+"_map.jpg", out[0][0]*255.0)
			cv2.imwrite(image[:-4]+"_clear.jpg", out[1][0]*255.0)
