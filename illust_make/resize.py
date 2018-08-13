import os
import glob
from PIL import Image,ImageFilter,ImageDraw,ImageOps
from tqdm import tqdm

files_jpeg = glob.glob('./solo+standing+1girl/*.jpeg')
files_png = glob.glob('./solo+standing+1girl/*.png')
files = files_jpeg+files_png
w=256
h=256


for f in tqdm(files):
    img = Image.open(f)
    img.thumbnail((256,256),Image.ANTIALIAS)

    bg = Image.new("RGBA", [w, h], (0, 0, 0, 0))
    bg.paste(img,((w-img.size[0])//2,(h-img.size[1])//2))
    f_name, ext = os.path.splitext(os.path.basename(f))
    # print(f_name)
    bg.save('./resized/'+f_name+'.png')
    bg_r = ImageOps.mirror(bg)
    bg_r.save('./resized/'+f_name+'_r.png')

