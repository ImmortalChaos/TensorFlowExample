"""
Face Detection Program
"""
import os
import argparse
import cv2


def get_filename(img_path):
    """
    get filename from image path
    """
    filename = os.path.splitext(img_path)
    return os.path.basename(filename[0])

def face_detection(img_path, show_marking, crop_face, target_path):
    """
    Face Detection from image
    """
    # Git : https://github.com/opencv/opencv/tree/master/data/haarcascades
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    filename = get_filename(img_path)
    detect_cnt = 0
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        if crop_face is False and show_marking:
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            if show_marking:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        if crop_face is True and target_path is not None:
            detect_cnt += 1
            target_filepath = os.path.join(target_path, filename+"_"+str(detect_cnt)+".png")
            print(target_filepath)
            cv2.imwrite(target_filepath, roi_color)
    if crop_face:
        return
    if target_path is not None:
        if detect_cnt:
            target_filepath = os.path.join(target_path, filename+"_result.png")
            print(target_filepath)
            cv2.imwrite(target_filepath, img)
    else:
        cv2.imshow('detect '+str(detect_cnt)+' people', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Detection Program')
    parser.add_argument('-source', type=str, required=True, default=os.getcwd(),
                        help='Appoint the source folder or file path')
    parser.add_argument('-target', type=str, default=None, help='Appoint the target folder')
    parser.add_argument('-show_marking', default=False, action='store_true',
                        help='Displays the found area.')
    parser.add_argument('-crop_face', default=False, action='store_true',
                        help='Saves the detected face area as an image.')
    args = parser.parse_args()
    source_folder = args.source
    face_detection(source_folder, args.show_marking, args.crop_face, args.target)
