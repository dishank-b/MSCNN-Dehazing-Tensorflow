# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import tensorflow as tf
import numpy as np
import cv2
from ops import *
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
			conv1 = Conv_2D(x, output_chan=5, kernel=[11,11], stride=[1,1], padding="SAME", train_phase=self.train_phase, 
							name="Conv1")
			# pool1 = max_pool(conv1, 2, 2, "max_pool1")
			# upsample1 = max_unpool(pool1, "upsample1")
			
			conv2 = Conv_2D(conv1, output_chan=5, kernel=[9,9], stride=[1,1], padding="SAME", train_phase=self.train_phase, 
							name="Conv2")
			# pool2 = max_pool(conv2, 2, 2, "max_pool2")
			# upsample2 = max_unpool(pool2, "upsample2")
			
			conv3 = Conv_2D(conv2, output_chan=10, kernel=[7,7], stride=[1,1], padding="SAME", train_phase=self.train_phase, 
							name="Conv3")
			# pool3 = max_pool(conv3, 2, 2, "max_pool3")
			# upsample3 = max_unpool(pool3, "upsample3")	

			linear = Conv_2D(conv3, output_chan=1, kernel=[1,1], stride=[1,1], padding="SAME", activation=tf.sigmoid, train_phase=self.train_phase, 
							add_summary=True, name="linear_comb")
			return linear

	def _fineNet(self, x, coarseMap):
		with tf.variable_scope("fineNet") as var_scope:
			conv1 = Conv_2D(x, output_chan=4, kernel=[7,7], stride=[1,1], padding="SAME", train_phase=self.train_phase, 
							name="Conv1")
			# pool1 = max_pool(conv1, 2, 2, "max_pool1")
			# upsample1 = max_unpool(pool1, "upsample1")
			
			concat = tf.concat([conv1, coarseMap], axis=3, name="CorMapConcat")

			conv2 = Conv_2D(concat, output_chan=5, kernel=[5,5], stride=[1,1], padding="SAME", train_phase=self.train_phase, 
							name="Conv2")
			# pool2 = max_pool(conv2, 2, 2, "max_pool2")
			# upsample2 = max_unpool(pool2, "upsample2")
			
			conv3 = Conv_2D(conv2, output_chan=10, kernel=[3,3], stride=[1,1], padding="SAME", train_phase=self.train_phase, 
							name="Conv3")
			# pool3 = max_pool(conv3, 2, 2, "max_pool3")
			# upsample3 = max_unpool(pool3, "upsample3")	

			linear = Conv_2D(conv3, output_chan=1, kernel=[1,1], stride=[1,1], padding="SAME", train_phase=self.train_phase, 
							add_summary=True, name="linear_comb")
			return linear

	def build_model(self):
		with tf.name_scope("Inputs") as scope:
			self.x = tf.placeholder(tf.float32, shape=[None,216,240,3], name="Haze_Image")
			self.y = tf.placeholder(tf.float32, shape=[None,216,240,1], name="TMap")
			self.train_phase = tf.placeholder(tf.bool, name="is_training")
			hazy_summ = tf.summary.image("Hazy image", self.x)
			map_summ = tf.summary.image("Trans Map", self.y)

		with tf.name_scope("Model") as scope:
			self.coarseMap = self._coarseNet(self.x)
			self.transMap = self._fineNet(self.x, self.coarseMap)

		with tf.name_scope("Loss") as scope:
			
			self.coarseLoss = tf.losses.mean_squared_error(self.y, self.coarseMap)
			self.fineLoss = tf.losses.mean_squared_error(self.y, self.transMap)
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
					in_images = train_imgs[0][itr:itr+batch_size][:,0,:,:,:]
					out_images = train_imgs[1][itr:itr+batch_size]

					sess_in = [self.coarse_solver, self.coarseLoss, self.merged_summ]
					coarse_out = self.sess.run(sess_in, {self.x:in_images, self.y:out_images, self.train_phase:True})
					self.train_writer.add_summary(coarse_out[2])

					sess_in  = [self.fine_solver, self.fineLoss, self.merged_summ]
					fine_out = self.sess.run(sess_in, {self.x:in_images, self.y:out_images, self.train_phase:True})
					self.train_writer.add_summary(fine_out[2])

					if itr%5==0:
						print "Epoch: ", epoch, "Iteration: ", itr, " Coarse Loss: ", coarse_out[1], "Fine Loss: ", fine_out[1], \
								"Loss: ", coarse_out[1]+fine_out[1]

				for itr in xrange(0, val_imgs[0].shape[0]-batch_size, batch_size):
					in_images = val_imgs[0][itr:itr+batch_size][:,0,:,:,:]
					out_images = val_imgs[1][itr:itr+batch_size]

					c_val_loss,f_val_loss ,summ = self.sess.run([self.coarseLoss,self.fineLoss ,self.merged_summ], {self.x: in_images, self.y: out_images,self.train_phase:False})
					self.val_writer.add_summary(summ)

					print "Epoch: ", epoch, "Iteration: ", itr, "Validation Loss: ", c_val_loss+f_val_loss

				if epoch%20==0:
					self.saver.save(self.sess, self.save_path+"MSCNN", global_step=epoch)
					print "Checkpoint saved"

					# random_img = train_imgs[0][np.random.randint(1, train_imgs[0].shape[0], 1)]

					# gen_imgs = self.sess.run([self.coarseMap], {self.x: random_img[:,0,:,:,:],self.train_phase:False})
					# print gen_imgs[0][0].shape
					# cv2.imwrite(self.output_path +str(epoch)+"_train_img.jpg", 255.0*gen_imgs[0][0])

	def test(self, input_imgs):
		sess=tf.Session()
		
		saver = tf.train.import_meta_graph(self.save_path+'MSCNN-240.meta')
		saver.restore(sess,tf.train.latest_checkpoint(self.save_path))

		graph = tf.get_default_graph()

		x = graph.get_tensor_by_name("Inputs/Haze_Image:0")
		is_train = graph.get_tensor_by_name("Inputs/is_training:0")
		y = graph.get_tensor_by_name("Model/fineNet/linear_comb/Relu:0")
		
		maps = sess.run(y, {x:input_imgs, is_train:False})

		return maps

