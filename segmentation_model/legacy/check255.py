import os
import glob
import numpy as np
from PIL import Image

src = '/mnt/raid5/_datasets/Breast/Breast_model_results/shandaer/tumor_mask'
paths = glob.glob(f'{src}/*.png')

for path in paths:
    im = Image.open(path)
    a = np.array(im)
    print(path)
    print(np.unique(a))
    print()
