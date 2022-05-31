# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 18:22:48 2020

@author: Manohar
"""

import face_recognition
import docopt 
from sklearn import svm
import os 
import pickle
def face_recognize(test):
	filename = 'svm_trained_nov_5.sav'
	clf = pickle.load(open(filename, 'rb'))
	# Load the test image with unknown faces into a numpy array 
	test_image = face_recognition.load_image_file(test) 

	# Find all the faces in the test image using the default HOG-based model 
	face_locations = face_recognition.face_locations(test_image)
	no = len(face_locations)

	# Predict all the faces in the test image using the trained classifier 
	for i in range(no):
		test_image_enc = face_recognition.face_encodings(test_image)[i] 
		name = clf.predict([test_image_enc]) 
		if(name =='hemasai'):
			return 4
		elif(name=='manohar'):
			return 2
		elif(name=='rohith'):
			return 1
		elif(name=='madhan'):
			return 3
		else:
			return 5
count = []
test_image = 'hem.jpg'
face_recognize(test_image)
def face_recognize2(dir,c): 
	# Training the SVC classifier 
	# The training data would be all the 
	# face encodings from all the known 
	# images and the labels are their names 
	encodings = [] 
	names = [] 
	# Training directory 
	if dir[-1]!='/': 
		dir += '/'
	train_dir = os.listdir(dir) 
	# Loop through each person in the training directory 
	for person in train_dir: 
			print("entered")
			pred = (face_recognize('10im/'+person))
			count.append(pred)
	return count
train_dir = '10im'
c= 0
val = face_recognize2(train_dir,c)
predval = max(set(val), key = val.count)
if predval==1:
    print('rohith')
#print(max(val))
