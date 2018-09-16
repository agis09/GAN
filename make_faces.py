import cv2
import os
from glob import glob
from tqdm import tqdm
data_dir = 'E:\\pict_make\\album'
out_face_path = 'E:\\pict_make\\face_add\\'
xml_path = "E:\\pict_make\\lbpcascade_animeface.xml"


def movie_to_image(num_cut):

    capture = cv2.VideoCapture(video_path)

    img_count = 0
    frame_count = 0

    while(capture.isOpened()):

        ret, frame = capture.read()
        if ret == False:
            break

        if frame_count % num_cut == 0:
            img_file_name = output_path + str(img_count) + ".jpg"
            cv2.imwrite(img_file_name, frame)
            img_count += 1

        frame_count += 1

    capture.release()

def face_detect(img_list):

    classifier = cv2.CascadeClassifier(xml_path)

    img_count = 51752
    for img_path in tqdm(img_list):

        org_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        face_points = classifier.detectMultiScale(gray_img, \
                scaleFactor=1.2, minNeighbors=2, minSize=(1,1))
        # print(img_path)
        name, _ = os.path.splitext(img_path[21:])
        # print(name)
        if int(name)>=9995 and int(name)<=11972:
            if int(name)%2 == 0:
                continue
        for points in face_points:
            x, y, width, height =  points
            tmp = min(y,org_img.shape[0]-(y+height),x,org_img.shape[1]-(x+width))
            add = min(tmp,height//3)
            dst_img = org_img[y-add:y+height+add, x-add:x+width+add]

            face_img = cv2.resize(dst_img, (128,128))
            new_img_name = out_face_path  + str(img_count) + '.png'
            # print(new_img_name)
            cv2.imwrite(new_img_name, face_img)
            img_count += 1
if __name__ == '__main__':

    # movie_to_image(int(10))

    images = glob(data_dir + "/*.jpg") + \
           glob(data_dir + "/*.png") + \
           glob(data_dir + "/*.jpeg") + \
           glob(data_dir + "/*.bmp")
    face_detect(images)