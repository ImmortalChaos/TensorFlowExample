import os
import argparse
import cv2

def faceDetection(imgPath, showMarking=False, targetPath=None) :
	# Git : https://github.com/opencv/opencv/tree/master/data/haarcascades
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

	img = cv2.imread(imgPath)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
	    if showMarking :
	        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	    roi_gray = gray[y:y+h, x:x+w]
	    roi_color = img[y:y+h, x:x+w]
	    eyes = eye_cascade.detectMultiScale(roi_gray)
	    for (ex,ey,ew,eh) in eyes:
	        if showMarking :
	            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	if targetPath is not None :
		cv2.imwrite(targetPath, img)
	else :		
	    cv2.imshow('img',img)
	    cv2.waitKey(0)
	    cv2.destroyAllWindows()

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='Face Detection Program')
	parser.add_argument('-source', type=str, required=True, default=os.getcwd(),
			help='Appoint the source folder or file path')
	parser.add_argument('-target', type=str, default=os.getcwd(),
			help='Appoint the target folder')
	parser.add_argument('-show_marking', default=False, action='store_true', 
			help='Displays the found area.')
	args = parser.parse_args()
	
	sourceFolder = args.source
	faceDetection(sourceFolder, args.show_marking, args.target)


