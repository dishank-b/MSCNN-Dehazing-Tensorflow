# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import tensorflow as tf
import numpy as np

def Conv_2D(x, output_chan, kernel=[5,5], stride=[2,2],padding="SAME" ,activation=tf.nn.relu, use_bn=False, train_phase=True,add_summary=False,name="Conv_2D"):
	input_shape = x.get_shape()
	kern = [kernel[0], kernel[1], input_shape[-1], output_chan]
	strd = [1, stride[0], stride[1], 1]
	with tf.variable_scope(name) as scope:
		W = tf.get_variable(name="W", shape=kern, initializer=tf.keras.initializers.he_normal())
		b = tf.get_variable(name="b", shape=[output_chan], initializer=tf.keras.initializers.he_normal())

		Conv2D = tf.nn.bias_add(tf.nn.conv2d(input=x, filter=W, strides=strd, padding=padding), b)
		
		if use_bn==True:
			Conv2D = Bn(Conv2D, is_train=train_phase)

		if activation!=None:
			out = activation(Conv2D)
		else:
			out = Conv2D
	
		if add_summary==True:
			weight_summ= tf.summary.histogram(name+"_W", W)  # Make sure you use summary.merge_all() if here you are adding the summaries
			bias_summ= tf.summary.histogram(name+"_b", b)
			if out.get_shape()[-1]<=3:
				feature_summ = tf.summary.image(name+"_feat", out)
		
		return out

def Dconv_2D(x, output_chan,batch_size ,kernel=[5,5], stride=[2,2], padding="SAME",activation=tf.nn.relu, use_bn=False, train_phase=True,add_summary=False, name="D_conv2D"):
	input_shape = x.get_shape().as_list()
	kern = [kernel[0], kernel[1], output_chan, input_shape[-1]]
	strd = [1, stride[0], stride[1], 1]
	output_shape = [batch_size,input_shape[1]*strd[1],input_shape[2]*strd[2],output_chan]
	with tf.variable_scope(name) as scope:
		W = tf.get_variable(name="W", shape=kern, initializer=tf.keras.initializers.he_normal())
		b = tf.get_variable(name="b", shape=[output_chan], initializer=tf.keras.initializers.he_normal())

		D_Conv2D = tf.nn.bias_add(tf.nn.conv2d_transpose(x, filter=W, output_shape=output_shape,strides=strd, padding=padding), b)
		
		if use_bn==True:
			D_Conv2D = Bn(D_Conv2D, is_train=train_phase)
			assert activation!=None
			out = activation(D_Conv2D)	

		else:
			if activation!=None:
				out= activation(D_Conv2D)
			else:
				out= D_Conv2D

		if add_summary==True:
			weight_summ= tf.summary.histogram(name+"_W", W)
			bias_summ= tf.summary.histogram(name+"_b", b)
			return out, [weight_summ, bias_summ]
		else:
			return out

def Dense(x, output_dim, use_bn=True, activation=tf.nn.relu, train_phase=True,add_summary=False, name="Dense"):
	input_dim = x.get_shape()[-1]
	with tf.variable_scope(name) as scope:
		W = tf.get_variable('W', shape=[input_dim, output_dim], initializer=tf.keras.initializers.he_normal())
		b = tf.get_variable('b', shape=[output_dim], initializer=tf.keras.initializers.he_normal())

		dense = tf.nn.bias_add(tf.matmul(x, W), b)

		if use_bn==True:
			dense = Bn(dense, is_train=train_phase)
			assert activation!=None
			out = activation(dense)
		else:
			if activation!=None:
				out= activation(dense)
			else:
				out = dense

		if add_summary==True:
			return out, [weight_summ, bias_summ]
		else:
			return out

def Bn(x, is_train=True):
	"""
	If you are not using update_collection=None here, then make sure to add
	control dependency of tf.GraphKeys.UPDATE_OPS before running optimizer op.
	"""
	return tf.contrib.layers.batch_norm(x, decay= 0.99, is_training=is_train, center= True, scale=True, reuse=False)

def L_Relu(x, alpha=0.1):
	return tf.maximum(x, alpha*x)

def BReLU(x, tmin=0.0, tmax=1.0):
	return tf.minimum(tmax, tf.maximum(tmin, x))

def L_BReLU(x, tmin=0.0, tmax=1.0, alpha=0.1):
	return tf.maximum(alpha*x, tf.minimum(x, tmax+alpha*(x-1)))

def max_pool(input, kernel=3, stride=2, name=None):
   """Max-pool

   Args:
      input : Input Tensor
      kernel: filter's width (= filter's height)
      stride: stride of the filter
      name  : Optional name for the operation

   Returns:
      Tensor after max-pool operation
   """
   if name is None: 
      name='max_pool'

   with tf.variable_scope(name):
      ksize = [1, kernel, kernel, 1]
      strides = [1, stride, stride, 1]
      output = tf.nn.max_pool(input, ksize=ksize, strides=strides,
         padding='SAME')
      return output

def make_mat(in_shape):
	a = np.zeros(shape=[in_shape[0]*2, in_shape[0]])
	index_0 = range(2*in_shape[0])
	index_1 = [int(i/2) for i in range(2*in_shape[0])]
	index_0 = np.array(index_0)
	index_1 = np.array(index_1)
	a[index_0, index_1] = 1.0
	return a

def make_mat_trans(in_shape):
	a = np.zeros(shape=[in_shape[0], in_shape[0]*2])
	index_1 = range(2*in_shape[0])
	index_0 = [int(i/2) for i in range(2*in_shape[0])]
	index_0 = np.array(index_0)
	index_1 = np.array(index_1)
	a[index_0, index_1] = 1.0
	return a

def max_unpool(value, name):
	with tf.variable_scope(name) as scope:
		in_shape = value.get_shape().as_list()
		assert in_shape[1]==in_shape[2]
		a = tf.py_func(make_mat, [in_shape[1]], tf.float32)
		b = tf.py_func(make_mat_trans, [in_shape[2]], tf.float32)
		out = tf.matmul(a, tf.matmul(input,b))
		print out.get_shape()
		return out