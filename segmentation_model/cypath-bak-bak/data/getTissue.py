from histomicstk.saliency.tissue_detection import (
            get_slide_thumbnail, get_tissue_mask)
import numpy as np
from PIL import Image
import os
import glob
import cv2

def getTissueMask_v1(im_path, max_mag, curr_mag):
    scale = max_mag / curr_mag
    im = Image.open(im_path)
    w,h = im.size
    w = w/scale
    h = h/scale
    w = int(w)
    h = int(h)
    #im = im.resize((w,h))
    im = np.array(im).astype(np.uint8)
    labeled, mask = get_tissue_mask(
                im, deconvolve_first=False,
                n_thresholding_steps=1, sigma=0., min_size=300)
    return mask.astype(np.uint8)

def getTissueMask(im_path):
    im = Image.open(im_path)
    im = np.array(im).astype(np.uint8)
    white = np.array([230,230,230])
    mask = np.all(im < white, axis=2)
    return mask.astype(np.uint8)

def linkTissueEpiThreshold(tissue_src, tissue_suffix, epi_mask_src, mask_suffix, ratio,tar_dir):
    src_paths = glob.glob(f'{tissue_src}/*{tissue_suffix}')
    for src_path in src_paths:
        name = src_path.split('/')[-1].replace(tissue_suffix, '')
        mask_path = f'{epi_mask_src}/{name}{mask_suffix}'
        #print(mask_path, os.path.exists(mask_path))
        tissue_mask = getTissueMask(src_path)
        epi_mask = cv2.imread(mask_path, 0)
        epi_mask = epi_mask / 255 if np.max(epi_mask)>1 else epi_mask
        r = np.sum(epi_mask) / np.sum(tissue_mask)
        if r > ratio:
            os.symlink(src_path, f'{tar_dir}/{name}{tissue_suffix}')
            os.symlink(mask_path, f'{tar_dir}/{name}{mask_suffix}')
if __name__ == '__main__':
    #im_path = '/mnt/raid10/_datasets/Oral/TMA/IHCHE/he/1_A-3_real_B.png'
    #mask = getTissueMask(im_path)
    #Image.fromarray(mask*255).save('/home/yxw1452/Desktop/tmp.png')
    tissue_src = '/raid10/_datasets/Oral/TMA/IHCHE/he'
    tissue_suffix = '_real_B.png'
    epi_mask_src = '/raid10/_datasets/Oral/TMA/IHCHE/mask_threshold_adjusted'
    mask_suffix = '_real_A_mask.png'
    ratio = .3
    tar_dir = '/raid10/_datasets/Oral/TMA/IHCHE/tissue_epi_0.3'
    linkTissueEpiThreshold(tissue_src, tissue_suffix, epi_mask_src, mask_suffix, ratio,tar_dir)

