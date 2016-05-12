import numpy as np
import cv2
import os, sys

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Open a file
#path = "C:\Users\michael\Documents\Python Scripts\Emotion data"
path = "/home/michael/Documents/School/neural networks/Data"
dirs = os.listdir( path )
#dirs = os.listdir( os.curdir )

check_string = "Rafd090"
images = np.array([])
images = images.reshape([-1,50*50])
labels = np.array([])
labels = labels.reshape([-1,8])

data_set = np.array([])
data_set = data_set.reshape([-1,50*50])
data_labels = np.array([])
data_labels = data_labels.reshape([-1,8])

test_set = np.array([])
test_set = test_set.reshape([-1,50*50])
test_labels = np.array([])
test_labels = test_labels.reshape([-1,8])

image_labels = np.array([])
image_labels = image_labels.reshape([-1,8])
directories = np.array([])
dist_contrast = np.array([])
dist_brightness = np.array([])
roi_color = np.array([])

count = 0
test_batch_size = 160*3
 
#while(True):
for files in dirs:
     # Capture frame-by-frame
     full_path = os.path.join(path, files) 
     #print full_path

     frontal = files.find(check_string)
     if frontal >= 0:
          directories = np.append(directories, files)
          count = count+1

print count

for good_files in directories:#[0:100]:
          full_path = os.path.join(path, good_files) 
          frame = cv2.imread(full_path) #cap.read()
      
          # Our operations on the frame come here
          gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

          faces = face_cascade.detectMultiScale(gray, 1.3, 5)
          for (x,y,w,h) in faces:
              #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
              roi_gray = gray[y:y+h, x:x+w]
              roi_color = frame[y:y+h, x:x+w]
              
       
          # Display the resulting frame
          #cv2.imshow('frame',frame)
          
          res = cv2.resize(roi_gray,(50, 50), interpolation = cv2.INTER_CUBIC)
          dist_contrast = res.copy()
          dist_brightness = res.copy()

          alpha = np.random.uniform(0.5,1.5)
          beta = np.random.uniform(0,100)
          cv2.multiply(dist_contrast,alpha, dist_contrast, scale=1) 
          dist_brightness  = cv2.add(dist_brightness,beta)

          
          if good_files.find("Rafd090_01") >=0:
               
               flat = res.flatten()/255.
               test_set = np.vstack([test_set, flat])
               flat = dist_contrast.flatten()/255.
               test_set = np.vstack([test_set, flat])
               flat = dist_brightness.flatten()/255.
               test_set = np.vstack([test_set, flat])
               #cv2.imshow('Face',res)

               if good_files.find("angry") >=0:
                    #labels = np.vstack([labels, [1, 0, 0, 0, 0, 0, 0, 0]])
                    image_labels = [1, 0, 0, 0, 0, 0, 0, 0]
               if good_files.find("contemptuous") >=0:
                    image_labels = [0, 1, 0, 0, 0, 0, 0, 0]
               if good_files.find("disgusted") >=0:
                    image_labels = [0, 0, 1, 0, 0, 0, 0, 0]
               if good_files.find("fearful") >=0:
                    image_labels = [0, 0, 0, 1, 0, 0, 0, 0]
               if good_files.find("happy") >=0:
                    image_labels = [0, 0, 0, 0, 1, 0, 0, 0]
               if good_files.find("neutral") >=0:
                    image_labels = [0, 0, 0, 0, 0, 1, 0, 0]
               if good_files.find("sad") >=0:
                    image_labels = [0, 0, 0, 0, 0, 0, 1, 0]
               if good_files.find("surprised") >=0:
                    image_labels = [0, 0, 0, 0, 0, 0, 0, 1]

               #print image_labels
                    
               test_labels = np.vstack([test_labels,image_labels])
               test_labels = np.vstack([test_labels,image_labels])
               test_labels = np.vstack([test_labels,image_labels])

          else:
               

               flat = res.flatten()/255.
               data_set = np.vstack([data_set, flat])
               flat = dist_contrast.flatten()/255.
               data_set = np.vstack([data_set, flat])
               flat = dist_brightness.flatten()/255.
               data_set = np.vstack([data_set, flat])
               #cv2.imshow('Face',res)

               if good_files.find("angry") >=0:
                    #labels = np.vstack([labels, [1, 0, 0, 0, 0, 0, 0, 0]])
                    image_labels = [1, 0, 0, 0, 0, 0, 0, 0]
               if good_files.find("contemptuous") >=0:
                    image_labels = [0, 1, 0, 0, 0, 0, 0, 0]
               if good_files.find("disgusted") >=0:
                    image_labels = [0, 0, 1, 0, 0, 0, 0, 0]
               if good_files.find("fearful") >=0:
                    image_labels = [0, 0, 0, 1, 0, 0, 0, 0]
               if good_files.find("happy") >=0:
                    image_labels = [0, 0, 0, 0, 1, 0, 0, 0]
               if good_files.find("neutral") >=0:
                    image_labels = [0, 0, 0, 0, 0, 1, 0, 0]
               if good_files.find("sad") >=0:
                    image_labels = [0, 0, 0, 0, 0, 0, 1, 0]
               if good_files.find("surprised") >=0:
                    image_labels = [0, 0, 0, 0, 0, 0, 0, 1]

               #print image_labels
                    
               data_labels = np.vstack([data_labels,image_labels])
               data_labels = np.vstack([data_labels,image_labels])
               data_labels = np.vstack([data_labels,image_labels])
     
          if cv2.waitKey(1) & 0xFF == ord('q'):
           break

#data_set = images[test_batch_size:,:]
#test_set = images[0:test_batch_size,:]

#data_labels = labels[test_batch_size:,:]
#test_labels = labels[0:test_batch_size,:]

np.save("data_set_extended.npy", data_set)
np.save("test_set_extended.npy", test_set)
np.save("data_labels_extended.npy", data_labels)
np.save("test_labels_extended.npy", test_labels)

print images
print data_set
print test_set
print data_labels
print test_labels
   
# When everything done, release the capture
cv2.destroyAllWindows()
