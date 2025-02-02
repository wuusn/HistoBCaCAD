import os
import glob
from PIL import Image
import numpy as np
from cypath.data.multiRun import multiRunStarmap

def diff_one(path1, path2, save_path):

    im1 = Image.open(path1)
    im2 = Image.open(path2)

    a1 = np.array(im1).astype(np.uint8)
    a2 = np.array(im2).astype(np.uint8)

    diff = a1-a2
    plus = diff == 255 # a1 has, a2 not has
    minus = diff == -255 # a2 has, a1 not has

    w,h = im1.size
    diff = np.zeros((h,w,3))
    red = np.array([230,0,0])
    green = np.array([0,152,0])
    diff[plus]=red
    diff[minus]=green

    diff = diff.astype(np.uint8)
    im = Image.fromarray(diff)
    im.save(save_path)

if __name__ == '__main__':
    cohorts = ['qingdao', 'shandaer']
    for cohort in cohorts:
        src1 = f'/mnt/raid5/_datasets/Breast/Breast_model_results/{cohort}/tumor_mask_morph'
        src2 = f'/mnt/raid5/_datasets/Breast/Breast_model_results/{cohort}/tumor_mask_5_127_morph_otsu_tissue'
        save_dir = f'/mnt/raid5/_datasets/Breast/Breast_model_results/{cohort}/tumor_mask_morph_vs_5_127_morph_otsu_tissue'
        os.makedirs(save_dir, exist_ok=True)
        path1s = sorted(glob.glob(f'{src1}/*.png'))
        path2s = [path1.replace(src1, src2) for path1 in path1s]
        save_paths = [path1.replace(src1, save_dir) for path1 in path1s]
        multiRunStarmap(diff_one, path1s, path2s, save_paths)

        

