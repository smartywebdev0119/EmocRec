from os import listdir
from os.path import isfile, join
import numpy as np
from constants import *
import cv2
import random

class DatasetLoader(object):

  def __init__(self):
    self._images = np.array([])
    self._labels = np.array([])
    self._epochs_completed = 0
    self._index_in_epoch = 0

  def load_from_save(self):
    self._images      = np.load(join(SAVE_DIRECTORY, SAVE_DATASET_IMAGES_FILENAME))
    self._labels      = np.load(join(SAVE_DIRECTORY, SAVE_DATASET_LABELS_FILENAME))
    self._images_test = np.load(join(SAVE_DIRECTORY, SAVE_DATASET_IMAGES_TEST_FILENAME))
    self._labels_test = np.load(join(SAVE_DIRECTORY, SAVE_DATASET_LABELS_TEST_FILENAME))
    self._images      = self._images.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
    self._images_test = self._images.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
    self._labels      = self._labels.reshape([-1, len(EMOTIONS)])
    self._labels_test = self._labels.reshape([-1, len(EMOTIONS)])

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def images_test(self):
    return self._images_test

  @property
  def labels_test(self):
    return self._labels_test

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

class ImageFile:
  def __init__(self, image_path):
    self.image_path = image_path
    label = self.classify_label(image_path)
    image = np.asarray(bytearray(open(join(DATASET_PATH, image_path)).read()), dtype = 'uint8')
    image = ImageFile.format_image(image)
    if image is None or label is None:
      self._image = None
      self._label = None
      return
    self._image = image
    self._label = label
    #cv2.imshow("Imagen", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
  @classmethod
  def format_image(self, image):
    if len(image.shape) > 2 and image.shape[2] == 3:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
      image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    faces = cv2.CascadeClassifier(CASC_PATH).detectMultiScale(
        image,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (20, 20),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    # None is we don't found an image
    if not len(faces) > 0:
      return None
    max_area_face = faces[0]
    for face in faces:
      if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
        max_area_face = face
    # Chop image to face
    face = max_area_face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
    # Resize image to network size
    try:
      image = cv2.resize(image, (SIZE_FACE, SIZE_FACE)) / 255.
    except Exception:
      print("[+] Problem during resize")
      return None
    # cv2.imshow("Lol", image)
    # cv2.waitKey(0)
    return image

  def classify_label(self, image_path):
    for index, emotion in enumerate(EMOTIONS):
      if emotion in image_path:
        tmp = np.zeros(len(EMOTIONS))
        tmp[index] = 1.0
        return tmp
    # This should not happen
    raise Exception('There is an extrange emotion in file ' + image_path)

  @property
  def image(self):
    return self._image

  @property
  def label(self):
    return self._label
