from __future__ import division, print_function, absolute_import
import re
import numpy as np
from dataset_loader import DatasetLoader
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

class MoodRecognition:
  SIZE_FACE = 56
  SAVE_DEFAULT_PATH = 'model.tfl'

  def __init__(self):
    self.build_network()
    dataset = DatasetLoader()

  def build_network(self):
    # Building 'AlexNet'
    # https://github.com/tflearn/tflearn/blob/master/examples/images/alexnet.py
    print('[+] Building CNN')
    network = input_data(shape=[None, 224, 224, 3])
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 17, activation='softmax')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    self.network = network


  def start_training(self):
    # Training
    print('[+] Training network')
    self.model = tflearn.DNN(self.network, tensorboard_verbose = 3)
    self.model.fit(
      { 'input': X },
      { 'target': Y },
      n_epoch = 20,
      validation_set = (
        { 'input': testX },
        { 'target': testY }
      ),
      snapshot_step = 100,
      show_metric = True,
      run_id = 'convnet_mnist'
    )

  def save_model(self):
    self.model.save(SAVE_DEFAULT_PATH)
    print('[+] Model saved at ' + SAVE_DEFAULT_PATH)

  def put_points_image(self, image, points):
    index = 0
    while index < len(points - 1):
      center = (int(points[index]), int(points[index + 1]))
      print(center)
      cv2.circle(image, center, 10, (0, 0, 255), 1)
      index += 2

if __name__ == "__main__":
  network = MoodRecognition()


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