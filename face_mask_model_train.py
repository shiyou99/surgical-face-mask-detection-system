#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the necessary packages
from  tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


# In[2]:


# initialize the initial learning rate, number of epochs,and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32


# ## Loading and pre-processing our training data

# In[3]:


#Our training data contained image with mask and without mask
imagePaths_mask = []
imagePaths_without_mask = []

for filename in os.listdir("data/with_mask"):
  imagePaths_mask.append(filename)
for filename in os.listdir("data/without_mask"):
  imagePaths_without_mask.append(filename)


# In[4]:


data = []
labels = []

for img in imagePaths_mask:

  label = 'with_mask'

  image = load_img("data/with_mask/"+img, target_size=(224,224))
  image = img_to_array(image)
  image = preprocess_input(image)

  data.append(image)
  labels.append(label)

for img in imagePaths_without_mask:

  label = 'without_mask'

  image = load_img("data/without_mask/"+img, target_size=(224,224))
  image = img_to_array(image)
  image = preprocess_input(image)

  data.append(image)
  labels.append(label)

# convert the data and labels to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)


# ## Performing one-hot encoding on the labels

# In[5]:


# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)


# In[6]:


plt.imshow(data[0])


# In[7]:


# split training and testing data
(trainX, testX, trainY, testY) = train_test_split(data, labels,
    test_size=0.20, stratify=labels, random_state=42)


# In[8]:


#Initialising ImageDataGenerator
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")


# In[9]:


# load the MobileNetV2 network and fine tuning it
baseModel = MobileNetV2(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))


# In[11]:


headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)


# ## Defining final model

# In[12]:


model = Model(inputs=baseModel.input, outputs=headModel)


# In[13]:


#freezing base model's layers
for layer in baseModel.layers:
    baseModel.trainable = False


# In[14]:


# compile our model
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])


# ## Fitting our model

# In[15]:


history = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS,
    verbose = 1)


# ## Save our model into .pb and .h5 format

# In[16]:


model.save("models")  # as .pb
model.save("models/face_model.h5")  # as .h5


# ## Model Evaluation

# In[17]:


accuracy = history.history['accuracy']
loss = history.history['loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.title('Training accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.title('Training loss')
plt.legend()

plt.show()

