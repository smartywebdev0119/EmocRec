from __future__ import division, print_function, absolute_import
import re
import numpy as np
from dataset_loader import DatasetLoader, ImageFile
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from constants import *
from os.path import isfile, join
import sys

class MoodRecognition:

  def __init__(self):
    self.dataset = DatasetLoader()

  def build_network(self):
    # Building 'AlexNet'
    # https://github.com/tflearn/tflearn/blob/master/examples/images/alexnet.py
    print('[+] Building CNN')
    self.network = input_data(shape = [None, SIZE_FACE, SIZE_FACE, 1])
    self.network = conv_2d(self.network, 32, 4, activation = 'relu')
    self.network = max_pool_2d(self.network, 2)
    self.network = conv_2d(self.network, 64, 3, activation = 'relu')
    self.network = conv_2d(self.network, 64, 3, activation = 'relu')
    self.network = max_pool_2d(self.network, 2)
    self.network = fully_connected(self.network, 512, activation = 'relu')
    self.network = dropout(self.network, 0.5)
    self.network = fully_connected(self.network, len(EMOTIONS), activation = 'softmax')
    mom = tflearn.Momentum(0.1, lr_decay = 0.1, decay_step = 16000, staircase = True)
    self.network = regression(self.network,
      optimizer = mom,
      loss = 'categorical_crossentropy',
      learning_rate = 0.001)
    self.model = tflearn.DNN(
      self.network,
      checkpoint_path = SAVE_DIRECTORY + '/alexnet_mood_recognition',
      max_checkpoints = 1,
      tensorboard_verbose = 2
    )

  def build_and_save_dataset(self):
    self.dataset.build()
    self.dataset.save()

  def load_saved_dataset(self):
    try:
      # IF file exists
      self.dataset.load_from_save()
      print('[+] Dataset found and loaded')
    except Exception as err:
      # If not, we build them
      print('[+] Dataset was not found, building')
      self.build_and_save_dataset()

  def start_training(self):
    self.load_saved_dataset()
    self.build_network()
    if self.dataset is None:
      self.load_saved_dataset()
    # Training
    print('[+] Training network')
    self.model.fit(
      self.dataset.images, self.dataset.labels,
      validation_set = 0.1,
      n_epoch = 10,
      batch_size = 100,
      shuffle = True,
      show_metric = True,
      snapshot_step = 200,
      snapshot_epoch = True,
      run_id = 'alexnet_mood_recognition'
    )

  def predict(self, image):
    image = ImageFile.format_image(image)
    if image is None:
      return None
    image = image.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
    return self.model.predict(image)

  def save_model(self):
    self.model.save(join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME))
    print('[+] Model trained and saved at ' + SAVE_MODEL_FILENAME)

  def load_model(self):
    self.model.load(join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME))
    print('[+] Model loaded from ' + SAVE_MODEL_FILENAME)


def show_usage():
  print('[!] Usage: python mood_recognition.py')
  print('\t mood_recognition.py build-dataset \t Build and saved dataset with labels')
  print('\t mood_recognition.py train-model \t Trains and saves model with saved dataset')
  print('\t mood_recognition.py poc \t Launch the proof of concept')

if __name__ == "__main__":
  if len(sys.argv) <= 1:
    show_usage()
    exit()

  network = MoodRecognition()
  if sys.argv[1] == 'build-dataset':
    network.build_and_save_dataset()
  elif sys.argv[1] == 'train-model':
    network.start_training()
    network.save_model()
  elif sys.argv[1] == 'poc':
    import poc
  else:
    show_usage()
