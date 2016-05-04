""" Convolutional network applied to CIFAR-10 dataset classification task.
References:
    Learning Multiple Layers of Features from Tiny Images, A. Krizhevsky, 2009.
Links:
    [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
"""
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import numpy as np

# Data loading and preprocessing
from tflearn.datasets import cifar10
#(X, Y), (X_test, Y_test) = cifar10.load_data()
X = np.load("data_set.npy")
X = np.reshape(X, [-1, 50, 50, 1])
X_test = np.load("test_set.npy")
X_test = np.reshape(X_test, [-1, 50, 50, 1])
Y = np.load("data_labels.npy")
Y_test = np.load("test_labels.npy")

X, Y = shuffle(X, Y)
#Y = to_categorical(Y, 8)
#Y_test = to_categorical(Y_test, 8)

# Convolutional network building
network = input_data(shape=[None, 50, 50, 1])
network = conv_2d(network, 50, 1, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 100, 1, activation='relu')
network = conv_2d(network, 100, 1, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 1250, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 8, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=100, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=50, run_id='cifar10_cnn')

model.save ("face_model")
