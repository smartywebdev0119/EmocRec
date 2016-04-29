# Face features detection using CNN - Helen Dataset

from __future__ import division, print_function, absolute_import
import re
import cv2
from cv2.cv import *
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from os import listdir
from os.path import isfile, join

ANNOTATIONS_PATH = './helen-datasets/annotation'
IMAGES_PATH = './helen-datasets/images'
SIZE_FACE = 56
CASC_PATH = 'haarcascade_frontalface_default.xml'
FACE_CASCADE = cv2.CascadeClassifier(CASC_PATH)

def put_points_image(image, points):
  index = 0
  while index < len(points - 1):
    center = (int(points[index]), int(points[index + 1]))
    print(center)
    cv2.circle(image, center, 10, (0, 0, 255), 1)
    index += 2

def load_image(text_file):
  f = open(join(ANNOTATIONS_PATH, text_file), 'r')
  image_name = f.readline().strip()
  image_path = join(IMAGES_PATH, image_name + '.jpg')
  output = np.hstack([line.strip().split(',') for line in f.readlines()])
  if not re.match('\d+_\d', image_name) or not isfile(image_path):
    return None
  output = np.asarray(output, dtype='|S4').astype(np.float)
  image = np.asarray(bytearray(open(image_path).read()), dtype='uint8')
  image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
  size_face = 50
  faces = FACE_CASCADE.detectMultiScale(
      image,
      scaleFactor = 1.1,
      minNeighbors = 5,
      minSize = (size_face, size_face),
      flags = cv2.cv.CV_HAAR_SCALE_IMAGE
  )
  if not len(faces) > 0:
    return None
  # Chop image
  face = faces[0]
  image = image[face[0]:(face[0] + face[2]), face[1]:(face[1] + face[3])]
  # Substract position face to points
  index = 0
  while index < output.size - 1:
    output[index] -= face[1]
    output[index + 1] -= face[0]
    index += 2
  # output = cv2.resize(image, (SIZE_FACE, SIZE_FACE))
  # Resize image
  # try:
  #  resized_image = cv2.resize(image, (SIZE_FACE, SIZE_FACE))
  #except Exception:
  #  return None
  put_points_image(image, output)
  cv2.imshow("Imagen", image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  return image, output


# Data loading and preprocessing

# X, Y, testX, testY = mnist.load_data(one_hot=True)

print("[+] Loading images:")
X = np.array([])
Y = np.array([])
testX = np.array([])
testY = np.array([])
text_files = [f for f in listdir(ANNOTATIONS_PATH) if isfile(join(ANNOTATIONS_PATH, f))]
for index, text_file in enumerate(text_files):
  loaded_image = load_image(text_file)
  if loaded_image is None:
    continue
  print("\t[-] Loaded image " + str(index))
  if index < 300:
    X = np.append(X, loaded_image[0])
    Y = np.append(Y, loaded_image[1])
  elif index < 330:
    testX = np.append(testX, loaded_image[0])
    testY = np.append(testY, loaded_image[1])
  else:
    break

X = X.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
testX = testX.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
Y = Y.reshape([-1, 388])
testY = testY.reshape([-1, 388])

print("[+] Building CNN")

# Building convolutional network
network = input_data(shape=[None, SIZE_FACE, SIZE_FACE, 1], name='input')
network = conv_2d(network, 32, 3, activation='relu', regularizer='L2')
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer='L2')
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 388, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='target')

# Training
print("[+] Training")
model = tflearn.DNN(network, tensorboard_verbose=3)
model.fit(
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
model.save("modelo1.tfl")
