#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import argparse
import cv2
import os
import time


# ### Load pretrained model

# In[4]:


model = tf.keras.models.load_model("models/face_model.h5")


# ### Loading face detection classifier

# In[5]:


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# In[7]:


cap = cv2.VideoCapture(0) # open a camera for video capturing.

while True:
    
    ret, img = cap.read()
    
    if ret == True:
        time.sleep(1/25) # add delay in the execution of the program

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # convert img(frame) from rgb color space to gray
        faces = face_cascade.detectMultiScale(gray, 1.3, 8) # (gray, 1.3, 8) => image, scaleFactor, minNeighbours
        # if faces are found, it returns the positions of detected faces as Rect(x,y,w,h). 
        
        for (x, y, w, h) in faces:

            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            (mask, withoutMask) = model.predict(face)[0]
            mask = mask*100 # accuracy of mask on
            withoutMask = withoutMask*100 # accuracy of no mask

            font = cv2.FONT_HERSHEY_SIMPLEX
            # get text size in pixel
            textSize = cv2.getTextSize(text="No Mask: " + str("%.2f" % round(mask, 2)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=3)

            # draw a text string and rectangle on the image
            if mask > withoutMask:
                cv2.putText(img,
                            text = "Mask: " + str("%.2f" % round(mask, 2)),
                            org = (x-5,y-20),
                            fontFace=font,
                            fontScale = (2*w)/textSize[0][0],
                            color = (0, 255, 0),
                            thickness = 3,
                            lineType = cv2.LINE_AA)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
            else:
                cv2.putText(img,
                            text = "No Mask: " + str("%.2f" % round(withoutMask, 2)),
                            org = (x-5,y-20),
                            fontFace=font,
                            fontScale = (1.8*w)/textSize[0][0],
                            color = (0, 0, 255),
                            thickness = 3,
                            lineType = cv2.LINE_AA)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # display    
        cv2.imshow("Face Mask Detection",img)

        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
    else:
        break

# when everything done, release the capture
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




