""" 
Usage: 
face_recognize.py -d <train_dir> -i <test_image> 

Options: 
-h, --help					 Show this help 
-d, --train_dir =<train_dir> Directory with 
								images for training 
-i, --test_image =<test_image> Test image 
"""

# importing libraries 
import face_recognition
import docopt 
from sklearn import svm
import os 
import pickle
def face_recognize(dir, test): 
	encodings = [] 
	names = [] 
	if dir[-1]!='/': 
		dir += '/'
	train_dir = os.listdir(dir)  
	for person in train_dir: 
		pix = os.listdir(dir + person) 
 
		for person_img in pix: 
			# Get the face encodings for the face in each image file 
			face = face_recognition.load_image_file( 
				dir + person + "/" + person_img) 
			face_bounding_boxes = face_recognition.face_locations(face) 

			# If training image contains exactly one face 
			if len(face_bounding_boxes) == 1: 
				face_enc = face_recognition.face_encodings(face)[0] 
				# Add face encoding for current image 
				# with corresponding label (name) to the training data 
				encodings.append(face_enc) 
				names.append(person) 
			else: 
				print(person + "/" + person_img + " can't be used for training") 

	# Create and train the SVC classifier 
	clf = svm.SVC(gamma ='scale',probability=True) 
	clf.fit(encodings, names)
    
	filename = 'svm_trained_model.sav'
	pickle.dump(clf, open(filename,'wb'))
        
def main(): 
	args = docopt.docopt(__doc__) 
	train_dir = args["--train_dir"] 
	test_image = args["--test_image"] 
	face_recognize(train_dir, test_image) 
    
    
if __name__=="__main__": 
	main() 