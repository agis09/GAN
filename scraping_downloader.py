import requests, os
from bs4 import BeautifulSoup
import cv2
from glob import glob
from tqdm import tqdm


out_face_path = 'E:\\pict_make\\face_add\\'
xml_path = "E:\\pict_make\\lbpcascade_animeface.xml"

save_dir = 'E:\\pict_make\\tmp'

target_url = 'http://e-shuushuu.net'
page_tag = '/?page='


def download_image(url, directory):
    '''Save an image from a URL to a new directory.'''
    image = requests.get(url)
    filetype = image.headers['content-type'].split('/')[-1]
    # print(url.split('/'))
    name = directory + "/" + url.split("/")[-1] + "." + filetype

    file = open(name, 'wb')
    file.write(image.content)
    file.close()


while True:
    page_count = 1
    url =  target_url + page_tag + str(page_count)
    req = requests.get(url)
    print("\n")
    print("loading..."+url)
    print(req)

    if req == None:
        print("Finished! Downloaded " + str(page_count) + " images!")
        break

    soup = BeautifulSoup(req.text, 'lxml')

    for a in soup.find_all("a", class_="thumb_image"):
        img_url = target_url + a.get('href')
        print('saving...' + img_url)
        download_image(img_url, save_dir)



