import os
import glob
from PIL import Image,ImageFilter,ImageDraw,ImageOps
from tqdm import tqdm
import warnings
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

files_jpeg = glob.glob('/home/dmitri/PycharmProjects/GAN/illust_make/solo+standing+1girl/*.jpeg')
files_png = glob.glob('/home/dmitri/PycharmProjects/GAN/illust_make/solo+standing+1girl/*.png')
files = files_jpeg+files_png
w=128
h=128


for f in tqdm(files):
    warnings.filterwarnings('error')
    try:
        img = Image.open(f)
        img.thumbnail((128, 128), Image.ANTIALIAS)

        bg = Image.new("RGBA", [w, h], (0, 0, 0, 0))
        bg.paste(img, ((w - img.size[0]) // 2, (h - img.size[1]) // 2))
        f_name, ext = os.path.splitext(os.path.basename(f))
        # print(f_name)
        bg.save('./dcgan/resized/solo+standing+1girl/' + f_name + '.png')
        bg_r = ImageOps.mirror(bg)
        bg_r.save('./dcgan/resized/solo+standing+1girl/' + f_name + '_r.png')
    except Warning:
        print("warning raised")
        continue


