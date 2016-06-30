from __future__ import division, print_function, absolute_import

import csv
from PIL import Image
import tflearn
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

# csv import settings
trainingsize = 20000
testsize = 2000
dimension = 48
file = 'fer2013.csv'
dataset = np.empty([0,dimension*dimension])
labelset = np.empty([0,7])

trainingset = np.load("train_set_fer2013.npy")
trainingset = np.reshape(trainingset, [-1, 48, 48, 1])
testset = np.load("valid_set_fer2013.npy")
testset = np.reshape(testset, [-1, 48, 48, 1])
trainingsetlabels = np.load("train_labels_fer2013.npy")
testsetlabels = np.load("valid_labels_fer2013.npy")

#X, Y = shuffle(X, Y)


#Network building
network = input_data(shape=[None,48,48,1])
network = conv_2d(network, 64, 5, activation= 'relu')
network = local_response_normalization(network)
network = max_pool_2d(network, 3, strides = 2)
network = conv_2d(network, 64, 5, activation = 'relu')
network = max_pool_2d(network, 3, strides = 2)
network = conv_2d(network, 128, 4, activation = 'relu')
network = dropout(network, 0.3)
network = fully_connected(network,3072,activation = 'relu')
network = fully_connected(network,7,activation = 'softmax')
network = regression(network, optimizer='momentum',loss = 'categorical_crossentropy',
      learning_rate = 0.001)

model = tflearn.DNN(network)
model.fit(
	trainingset, trainingsetlabels,
	validation_set = (testset,testsetlabels),
	n_epoch = 40,
	shuffle = True,
	show_metric=True,
	run_id = 'Gudi_mood_recognition'
)
model.save ("Gudi_model")

