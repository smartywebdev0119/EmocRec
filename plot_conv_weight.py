from __future__ import division, absolute_import
import re
import numpy as np
from dataset_loader import DatasetLoader
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from constants import *
from os.path import isfile, join
import random
import cv2
import sys
import matplotlib.pyplot as plt

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def build_network():
  # Smaller 'AlexNet'
  # https://github.com/tflearn/tflearn/blob/master/examples/images/alexnet.py
  print('[+] Building CNN')
  network = input_data(shape = [None, SIZE_FACE, SIZE_FACE, 1])
  conv_1 = conv_2d(network, 96, 11, strides = 4, activation = 'relu')
  network = max_pool_2d(conv_1, 3, strides = 2)
  network = local_response_normalization(network)
  network = conv_2d(network, 256, 5, activation = 'relu')
  network = max_pool_2d(network, 3, strides = 2)
  network = local_response_normalization(network)
  network = conv_2d(network, 256, 3, activation = 'relu')
  network = max_pool_2d(network, 3, strides = 2)
  network = local_response_normalization(network)
  network = fully_connected(network, 1024, activation = 'tanh')
  network = dropout(network, 0.5)
  network = fully_connected(network, 1024, activation = 'tanh')
  network = dropout(network, 0.5)
  network = fully_connected(network, len(EMOTIONS), activation = 'softmax')
  network = regression(network,
    optimizer = 'momentum',
    loss = 'categorical_crossentropy')
  model = tflearn.DNN(
    network,
    checkpoint_path = SAVE_DIRECTORY + '/alexnet_mood_recognition',
    max_checkpoints = 1,
    tensorboard_verbose = 2
  )
  if isfile(join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME)):
    model.load(join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME))
    print('[+] Model loaded from ' + SAVE_MODEL_FILENAME)
  return conv_1.W

img = deprocess_image(build_network())
cv2.imshow("Lol", img)