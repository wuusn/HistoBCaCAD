src_dir = '/raid5/10x512NormalPatchOut'
src_ext = '.png'

import glob
import numpy as np
from PIL import Image
import os
import cv2

def one(path):
    im = Image.open(path)
    np_im = np.array(im)
    np_im = np_im / 255 if np.max(np_im)>1 else np_im
    np_im.astype(np.uint8)
    cv2.imwrite(path, np_im)

src_paths = glob.glob(f'{src_dir}/*/*{src_ext}')
from cypath.data.multiRun import multiRunStarmap

multiRunStarmap(one, src_paths)
