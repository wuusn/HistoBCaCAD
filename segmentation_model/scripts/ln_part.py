import glob
import os
import random
import shutil

size = 10
src_dir = '/ssd/Breast/split_L1_10x512_mask_Region9918_n2/test'
src_ext = '.jpg'
mask_ext = '.png'
tar_dir = '/ssd/Breast/split_L1_10x512_mask_Region9918_n2/small_test'

src_paths = glob.glob(f'{src_dir}/*{src_ext}')
random.shuffle(src_paths)
src_paths = src_paths[:size]

for src_path in src_paths:
    mask_path = src_path.replace(src_ext, mask_ext)
    shutil.copyfile(src_path, src_path.replace(src_dir, tar_dir))
    shutil.copyfile(mask_path, mask_path.replace(src_dir, tar_dir))


