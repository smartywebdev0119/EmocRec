# Proof-of-concept
import tflearn
import cv2
import sys
from constants import *
from mood_recognition import MoodRecognition
import numpy as np

# Load Model
network = MoodRecognition()
network.build_network()
try:
    network.load_model()
except Exception as err:
    print('[!] Saved model not found, exit(): ')
    print(err)
    exit()

video_capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

feelings_faces = []
for index, emotion in enumerate(EMOTIONS):
    feelings_faces.append(cv2.imread('./emojis/' + emotion + '.png', 1))

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Predict result with network
    result = network.predict(frame)

    # Draw face in frame

    # Write results in frame
    if result is not None:
      for index, emotion in enumerate(EMOTIONS):
        cv2.putText(frame, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 0.5, (0,255,0), 1);
        cv2.rectangle(frame, (130, index * 20 + 10), (130 + int(result[0][index] * 100), (index + 1) * 20 + 4), (255, 0, 0), -1)

      emotion_index = result[0].index(max(result[0]))
      frame[200:320, 10:130, :] = feelings_faces[emotion_index]

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()