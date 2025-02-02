import os
import glob
from PIL import Image

src_dir = '/raid10/_datasets/Breast/QiLu/10xRegion9918'
mask_paths = glob.glob(f'{src_dir}/*.png')
for mask_path in mask_paths:
    mask = Image.open(mask_path)
    im_path = mask_path.replace('.png', '.jpg')
    im = Image.open(im_path)
    if mask.size!=im.size:
        print(mask_path)
        size = im.size
        mask = mask.resize(size)
        mask.save(mask_path)
