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

class MoodRecognition:

  def __init__(self):
    self.build_network()

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
      checkpoint_path = 'saves/alexnet_mood_recognition',
      max_checkpoints = 1,
      tensorboard_verbose = 2
    )


  def start_training(self):
    self.dataset = DatasetLoader()
    # Training
    print('[+] Training network')
    self.model.fit(
      self.dataset.images, self.dataset.labels,
      validation_set = 0.1,
      n_epoch = 20,
      batch_size = 128,
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
    self.model.save(SAVE_DEFAULT_PATH)
    print('[+] Model trained and saved at ' + SAVE_DEFAULT_PATH)

  def load_model(self):
    self.model.load(SAVE_DEFAULT_PATH)
    print('[+] Model loaded from ' + SAVE_DEFAULT_PATH)

if __name__ == "__main__":
  network = MoodRecognition()
  network.start_training()
  network.save_model()


# print("[+] Loading images:")
# X = np.array([])
# Y = np.array([])
# testX = np.array([])
# testY = np.array([])
# text_files = [f for f in listdir(ANNOTATIONS_PATH) if isfile(join(ANNOTATIONS_PATH, f))]
# for index, text_file in enumerate(text_files):
#   loaded_image = load_image(text_file)
#   if loaded_image is None:
#     continue
#   print("\t[-] Loaded image " + str(index))
#   if index < 300:
#     X = np.append(X, loaded_image[0])
#     Y = np.append(Y, loaded_image[1])
#   elif index < 330:
#     testX = np.append(testX, loaded_image[0])
#     testY = np.append(testY, loaded_image[1])
#   else:
#     break

# X = X.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
# testX = testX.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
# Y = Y.reshape([-1, 388])
# testY = testY.reshape([-1, 388])