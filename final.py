import boto3
import pandas as pd
from firebase import firebase
import cv2
import os
import time
import face_recognition
import docopt 
from sklearn import svm
import pickle
import joblib

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
firebase = firebase.FirebaseApplication('https://icps-9cc0a.firebaseio.com/', None)
s3 = boto3.resource(
    service_name='s3',
    region_name='us-east-2',
    aws_access_key_id='AKIAWI34OZCUF7DLIFO3',
    aws_secret_access_key='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'
)



def face_recognize(test):
	filename = 'svm_trained_model.sav'
	clf = pickle.load(open(filename, 'rb'))
	# Load the test image with unknown faces into a numpy array 
	test_image = face_recognition.load_image_file(test) 
	face_locations = face_recognition.face_locations(test_image)
	no = len(face_locations)
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
        
        
        
        
def face_recognize2(dir): 
	count2 = [] 
	encodings = [] 
	names = [] 
	if dir[-1]!='/': 
		dir += '/'
	train_dir = os.listdir(dir)  
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




def upload_photo(dir):
    count = 0
    for obj in s3.Bucket('rohithalto').objects.all():
        count = count+1
    data =  { 'val': count  
             }
    result = firebase.post('/count/',data)
    print(result)
    s3.Bucket('rohithalto').upload_file(Filename=dir, Key=str(count)+'.png')
    
    
    
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
        temp_name=str(count1)+'.png'
        file_name_path = os.path.join(path_temp, temp_name)
        cv2.imwrite(file_name_path, face)
        cv2.putText(face, str(count1), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropper', face)
        time.sleep(0.1)
    else:
        print("Face not found")
        pass
    if cv2.waitKey(1) == 13 or count1 == 10:
        break
cap.release()
cv2.destroyAllWindows()
print("Collecting Samples Complete")
val = face_recognize2('test/'+i)
predval = max(set(val), key = val.count)


name=0
temp_file='5.png'
upload_photo_path=os.path.join(path_temp, temp_file)



if predval==1:
    print('rohith')
    name='rohith'
elif predval==2:
    print('manohar')
    name='manohar'
elif predval==3:
    print('madhan')
    name='madhan'
elif predval==4:
    print('hemasai')
    name='hemasai'
else :
    print('unknown')
    name='unknown'
    upload_photo(upload_photo_path)
print(val)




now_time=str(result.tm_hour)+":"+str(result.tm_min)
data =  { 'Name': name,
          'time': now_time,
          'access': "yes",
          'message':"permission accepted"
          }
result = firebase.post('/room1/',data)

