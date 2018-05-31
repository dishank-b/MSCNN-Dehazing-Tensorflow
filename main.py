# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import tensorflow as tf
import numpy as np
import glob
import sys
from models import *
import yaml
import os

####### Reading Hyperparameters #####
with open("config.yaml") as file:
	data = yaml.load(file)
	training_params = data['training_params']
	learning_rate = float(training_params['learning_rate'])
	batch_size = int(training_params['batch_size'])
	epoch_size = int(training_params['epochs'])
	data_path = training_params['data_path']
	model_params= data['model_params']
	descrip = model_params['descrip']
	log_dir = model_params['log_dir']
	if len(descrip)==0:
		raise ValueError, "Please give a proper description of the model you are training."


######## Making Directory #########
model_path = log_dir+sys.argv[1]
print "Model Path: ", model_path
if not os.path.exists(model_path):
	os.makedirs(model_path)
	os.makedirs(model_path+"/results")
	os.makedirs(model_path+"/tf_graph")
	os.makedirs(model_path+"/saved_model")


######### Loading Data ###########
Train_img = np.load(data_path+"Train.npy") # First image of each pair is clear image and
Val_img = np.load(data_path+"Val.npy")	   # second image is hazed images. 2nd image is input to the model
print "Data Loaded"
Train_img = 1/255.0*Train_img # Value scaled to [0,1]
Val_img = 1/255.0*Val_img

os.system('cp config.yaml '+model_path+'/config.yaml')

nnet = MSCNN(model_path)
nnet.build_model()
print "Model Build......"
# DD.train_model(Train_img, Val_img,learning_rate, batch_size, epoch_size)

