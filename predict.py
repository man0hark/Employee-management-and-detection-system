# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 15:20:35 2020

@author: Manohar
"""
from keras.preprocessing.image import load_img
from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import random
import cv2
from keras.models import load_model
from tensorflow.keras.models import load_model
import numpy as np
import pickle
from keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
model = load_model('facefeatures_new_model.h5')
image =load_img('C:/Users/Manohar/Desktop/project/35.jpg')
input_arr = img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = model.predict(input_arr)



# image=load_img('C:/Users/Manohar/Desktop/project/35.jpg')
# img_array = np.array(image)
# #Our keras model used a 4D tensor, (images x height x width x channel)
# #So changing dimension 128x128x3 into 1x128x128x3 
# img_array = np.expand_dims(img_array, axis=0)
# pred = model.predict(img_array)
# print(pred)
# print(pred.shape)