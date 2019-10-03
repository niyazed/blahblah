#!/usr/bin/env python
# coding: utf-8

# In[5]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 # opencv
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot as plt
from keras.models import load_model
from PIL import Image
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir())

# Any results you write to the current directory are saved as output.


# In[4]:


data = np.load('data/face-image.npy')
label = np.load('data/face-label.npy')

print("Data Samples: ", data.shape, "label samples :", label.shape)


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.10, random_state=42)


# In[7]:


print("Train: ", X_train.shape, "Train Labels", y_train.shape)
print("Train: ", X_test.shape, "Train Labels", y_test.shape)


# In[8]:


# save and compress the dataset for further use
np.savez_compressed('faces.npz', X_train, y_train, X_test, y_test)


# In[12]:


# load the face dataset
data = np.load('faces.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)


# In[ ]:


# load the facenet model
facenet_model = load_model('models/facenet_keras.h5')
print('Loaded Model')
facenet_model.summary()


# In[ ]:




