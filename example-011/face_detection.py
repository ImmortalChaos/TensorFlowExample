import os
import argparse
import cv2

def faceDetection(imgPath) :
	# Git : https://github.com/opencv/opencv/tree/master/data/haarcascades
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

	img = cv2.imread(imgPath)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
	    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	    roi_gray = gray[y:y+h, x:x+w]
	    roi_color = img[y:y+h, x:x+w]
	    eyes = eye_cascade.detectMultiScale(roi_gray)
	    for (ex,ey,ew,eh) in eyes:
	        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

	cv2.imshow('img',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='Face Detection Program')
	parser.add_argument('-source', type=str, required=True, default=os.getcwd(),
			help='Appoint the source folder or file path')
	args = parser.parse_args()
	
	sourceFolder = args.source
	faceDetection(sourceFolder)


