from .compareMasks import *
from .multiRun import *
import cv2
import os
import glob

from .evalModelResults import getPathsBySuffix

def saveOneMaskDifferredResultByFilePaths(gt_path, gt_suffix, p_path, save_dir):
    name = gt_path.split('/')[-1].replace(gt_suffix, '')
    gt = cv2.imread(gt_path, 0)
    p = cv2.imread(p_path, 0)
    fp = maskDifferFP(gt, p)
    fn = maskDifferFN(gt, p)
    cv2.imwrite(f'{save_dir}/{name}_fp.png', fp*255)
    cv2.imwrite(f'{save_dir}/{name}_fn.png', fn*255)

def saveMultiMaskDifferredResultsByDir(gt_dir, gt_suffix, p_dir, p_suffix, save_dir):
    gt_paths = getPathsBySuffix(gt_dir, gt_suffix)
    p_paths = getPathsBySuffix(p_dir, p_suffix)
    os.makedirs(save_dir, exist_ok=True)
    return  multiRunStarmap(saveOneMaskDifferredResultByFilePaths, gt_paths, gt_suffix, p_paths, save_dir)
