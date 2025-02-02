import os
import glob
import sys
from cypath.data.multiRun import getAllFiles, getAllFilesBySuffix
from cypath.data.multiRun import multiRunStarmap
import numpy as np
from PIL import Image

src_dir = '/raid5/10x512NormalPatchOnInOut'
paths = getAllFilesBySuffix(src_dir, '.jpg')
zeros = np.zeros((512,512)).astype(np.uint8)
im = Image.fromarray(zeros)
def one(path):
    im.save(path.replace('.jpg', '.png'))
multiRunStarmap(one, paths)




