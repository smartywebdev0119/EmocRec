from __future__ import division, print_function, absolute_import
import numpy as np
import cv2

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


 
cap = cv2.VideoCapture(0)

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

model.load ("face_model")

Y = np.array([])

test = 7

print("face_model" )
 
while(True):
     # Capture frame-by-frame
     ret, frame = cap.read()
     roi_color = frame[0:10, 0:10]
 
     # Our operations on the frame come here
     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
     for (x,y,w,h) in faces:
         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
         roi_gray = gray[y:y+h, x:x+w]
         roi_color = frame[y:y+h, x:x+w]
         #eyes = eye_cascade.detectMultiScale(roi_gray)
         #for (ex,ey,ew,eh) in eyes:
          #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
  
     # Display the resulting frame
     cv2.imshow('frame',frame)
     
     res = cv2.resize(roi_gray,(50, 50), interpolation = cv2.INTER_CUBIC)
     X = res/255.
     X = np.reshape(X, [-1, 50, 50, 1])
     Y = model.predict (X)
     
     print(Y[0][4])
     
     
     cv2.imshow('Face',res)
     
     if cv2.waitKey(1) & 0xFF == ord('q'):
      break
   
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
