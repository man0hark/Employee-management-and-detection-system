# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 15:25:49 2020

@author: Manohar
"""
import cv2
import os
import time
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def face_extractor(img):
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    if faces is ():
        return None
    for (x, y, w, h) in faces:
        x = x - 10
        y = y - 10
        cropped_face = img[y:y + h + 50, x:x + w + 50]
    return cropped_face
path='C:/Users/Manohar/Desktop/test'
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
count = 0
while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (400, 400))
        temp_name=str(count)+'.jpg'
        file_name_path = os.path.join(path_temp, temp_name)
        cv2.imwrite(file_name_path, face)
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropper', face)
        time.sleep(0.1)
    else:
        print("Face not found")
        pass
    if cv2.waitKey(1) == 13 or count == 10:  # 13 is the Enter Key
        break
cap.release()
cv2.destroyAllWindows()
print("Collecting Samples Complete")