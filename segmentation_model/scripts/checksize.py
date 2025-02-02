import glob
import os
from PIL import Image

im_paths = glob.glob(f'/raid10/_datasets/Breast/QiLu/10xRegion9918/*.jpg')
#im_paths = glob.glob(f'/raid10/_datasets/Breast/QiLu/20xRegion9918/*.jpg')
for path in im_paths:
    im = Image.open(path)
    mask_path = path.replace('jpg', 'png')
    mask = Image.open(mask_path)

    if im.size != mask.size:
        print(path)
        print(im.size, mask.size)
