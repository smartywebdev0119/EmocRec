# Proof-of-concept
import tflearn
import cv2
import sys
from constants import *
from mood_recognition import MoodRecognition
import numpy as np

# Load Model
network = MoodRecognition()
#network.load_model()

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Predict result with network
    result = network.predict(frame)

    # Draw face in frame

    # Write results in frame
    if result is not None:
      for index, emotion in enumerate(EMOTIONS):
        str_tmp = emotion + ' ' + str(result[0][index])
        cv2.putText(frame, str_tmp, (10, index * 15 + 30), cv2.FONT_HERSHEY_PLAIN, 1,(0,255,0),2);


    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()