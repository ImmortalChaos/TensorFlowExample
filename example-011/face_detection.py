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

def get_string_val(param, str_dict, num_value, default_value):
	for idx in range(len(str_dict)):
	    if str_dict[idx] in param:
	        return num_value[idx]
	return default_value

def get_parameter_in_filename(img_path):
    """
    Parsing parameter
    """
    scale_factor = get_string_val(img_path,
        ["_1.2f", "_1.25f", "_1.3f", "_1.4f", "_1.5f", "_1.6f", "_1.7f", "_1.8f"],
        [1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
        1.3)

    answer_cnt = get_string_val(img_path,
        ["_1p", "_2p", "_3p", "_4p", "_5p", "_6p", "_7p", "_8p", "_9p"],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        0)

    min_neighbors = get_string_val(img_path,
        ["_1n", "_2n", "_3n", "_4n", "_5n", "_6n", "_7n", "_8n"],
        [1, 2, 3, 4, 5, 6, 8],
        5)

    crop_index = get_string_val(img_path,
        ["_1i", "_2i", "_3i", "_4i", "_5i", "_6i", "_7i", "_8i", "_9i"],
        [1, 2, 3, 4, 5, 6, 8, 9],
        0)

    return scale_factor, min_neighbors, answer_cnt, crop_index

def face_detection(img_path, show_marking, crop_face, target_path):
    """
    Face Detection from image
    """
    # Git : https://github.com/opencv/opencv/tree/master/data/haarcascades
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    filename = get_filename(img_path)
    detect_cnt = 0
    scale_factor, min_neighbors, answer_cnt, crop_index = get_parameter_in_filename(img_path)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
    if answer_cnt and answer_cnt is not len(faces):
    	return 0
    faces_sorted = sorted(faces, key=lambda face: face[0])
    for (x, y, w, h) in faces_sorted:
        detect_cnt += 1
        if crop_index and crop_index is not detect_cnt:
        	continue
        if crop_face is False and show_marking:
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            if show_marking:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        if crop_face is True and target_path is not None:
            target_filepath = os.path.join(target_path, filename+"_"+str(detect_cnt)+".png")
            print(target_filepath)
            cv2.imwrite(target_filepath, roi_color)
    if crop_face:
        return detect_cnt
    if target_path is not None:
        if detect_cnt:
            target_filepath = os.path.join(target_path, filename+"_result.png")
            print(target_filepath)
            cv2.imwrite(target_filepath, img)
    else:
        cv2.imshow('detect '+str(detect_cnt)+' people(sf:'+str(scale_factor)+')', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return detect_cnt

def detect_images(folder_path, show_marking, crop_face, target_path):
    """
    detection from sub-folders
    """
    for (paths, dirs, files) in os.walk(folder_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext in ('.png', '.jpg'):
                detect_path = os.path.join(paths, filename)
                if face_detection(detect_path, show_marking, crop_face, target_path) == 0:
                    print(detect_path + "(detection fail!)")

def check_valid_argument(arg_val):
    """
    check validation
    """
    if not os.path.exists(arg_val.source):
        print("Can't find", arg_val.source)
        return False
    return True

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
    if check_valid_argument(args):
        if os.path.isdir(source_folder):
            detect_images(source_folder, args.show_marking, args.crop_face, args.target)
        elif face_detection(source_folder, args.show_marking, args.crop_face, args.target) == 0:
            print("Can't find face!")
