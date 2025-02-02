src_dir = '/ssd/Breast/split_L1_10x512_mask_Region9918_n2/small_test'
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
    #im = Image.fromarray(np_im)
    #im.save(path)
    cv2.imwrite(path, np_im)

src_paths = glob.glob(f'{src_dir}/*{src_ext}')
from cypath.data.multiRun import multiRunStarmap

multiRunStarmap(one, src_paths)
