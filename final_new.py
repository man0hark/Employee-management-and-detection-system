# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 16:14:52 2020

@author: Sowrappa
"""

import cv2
import os
import time
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# importing libraries 
import face_recognition
import docopt 
from sklearn import svm
import pickle
import boto3
import pandas as pd
from firebase import firebase

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
firebase = firebase.FirebaseApplication('https://icps-9cc0a.firebaseio.com/', None)
s3 = boto3.resource(
    service_name='s3',
    region_name='us-east-2',
    aws_access_key_id='AKIAWI34OZCUF7DLIFO3',
    aws_secret_access_key='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'
)

import datetime

x = datetime.datetime.now()
secondval=time.time()
result2=time.localtime(secondval)

def upload_photo(dir):
    count = 0
    for obj in s3.Bucket('rohithalto').objects.all():
        count = count+1
    data =  { 'val': count+1  
             }
    data2 = {'date': str(x.day) +'-'+ str(x.month) +'-'+str(x.year),
             'time': str(result2.tm_hour)+":"+str(result2.tm_min),
             'room':2
        
    }
    result = firebase.post('/count/',data)
    res = firebase.post('/unknown/',data2)
    print(result)
    s3.Bucket('rohithalto').upload_file(Filename=dir, Key=str(count+1)+'.png')
    
def face_recognize(test):
	filename = 'svm_trained_lessdata.sav'
	clf = pickle.load(open(filename, 'rb'))
	# Load the test image with unknown faces into a numpy array 
	test_image = face_recognition.load_image_file(test) 

	# Find all the faces in the test image using the default HOG-based model 
	face_locations = face_recognition.face_locations(test_image)
	no = len(face_locations)

	if no==1:
		for i in range(no):
			test_image_enc = face_recognition.face_encodings(test_image)[i] 
			name = clf.predict([test_image_enc]) 
			predmat = (clf.predict_proba([test_image_enc]))
			#print((predmat[0][3]))
			if (predmat[0][3])>0.70:
				return 1
			elif (predmat[0][0])>0.70:
				return 4
			elif (predmat[0][2])>0.70:
				return 2
			elif (predmat[0][1])>0.70:
				return 3
			else:
				return 5
def face_recognize2(dir): 
	count2 = []
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
			pred = (face_recognize(dir+person))
			count2.append(pred)
	return count2

def face_extractor(img):
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    if faces is ():
        return None
    for (x, y, w, h) in faces:
        x = x - 10
        y = y - 10
        cropped_face = img[y:y + h + 50, x:x + w + 50]
    return cropped_face
def upload_known(nam,my,nt):
    yx = datetime.datetime.now()
    data =  { 
            'Name': nam,
            'time': nt,
            'access': str(yx.day) +'-'+ str(yx.month) +'-'+str(yx.year),
            'message':my
            }
    result = firebase.post('/room1/',data)
path='test/'
second=time.time()
result=time.localtime(second)
i=str(result.tm_mon)+"-"+str(result.tm_mday)+"-"+str(result.tm_hour)+"-"+str(result.tm_min)+"-"+str(result.tm_sec)
print(i)
path_temp=path = os.path.join(path,i) 
try:
    os.mkdir(path_temp)
except OSError:
        print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)
cap = cv2.VideoCapture(0)
count1 = 0
while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count1 += 1
        face = cv2.resize(face_extractor(frame), (400, 400))
        temp_name=str(count1)+'.jpg'
        file_name_path = os.path.join(path_temp, temp_name)
        cv2.imwrite(file_name_path, face)
        cv2.putText(face, str(count1), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropper', face)
        time.sleep(0.1)
    else:
        print("Face not found")
        pass
    if cv2.waitKey(1) == 13 or count1 == 10:  # 13 is the Enter Key
        break
cap.release()
cv2.destroyAllWindows()
print("Collecting Samples Complete")

val = face_recognize2('test/'+i)
predval = max(set(val), key = val.count)
temp_file='5.jpg'
upload_photo_path=os.path.join(path_temp, temp_file)
if predval==1:
    print('rohith')
    name='rohith'
    mynew = "Accepted"
    now_time=str(result.tm_hour)+":"+str(result.tm_min)
    upload_known(name,mynew,now_time)
elif predval==2:
    print('manohar')
    name='manohar'
    mynew = "Accepted"
    now_time=str(result.tm_hour)+":"+str(result.tm_min)
    upload_known(name,mynew,now_time)

elif predval==3:
    print('madhan')
    name='madhan'
    mynew = "Accepted"
    now_time=str(result.tm_hour)+":"+str(result.tm_min)
    upload_known(name,mynew,now_time)

elif predval==4:
    print('hemasai')
    name='hemasai'
    mynew = "Accepted"
    now_time=str(result.tm_hour)+":"+str(result.tm_min)
    upload_known(name,mynew,now_time)
else :
    print('unknown')
    name='unknown'
    mynew = "-"
    upload_photo(upload_photo_path)
print(val)