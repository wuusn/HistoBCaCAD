import os
import cv2
import numpy as np
import geojson
#from post_processing import *

def mask2geojson(mask, upscale):
    bimg, contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cnts=[]
    for cnt in contours:
        if len(cnt)<4: continue
        arclen = cv2.arcLength(cnt, True)
        epsilon = arclen * 0.0002
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cnt=approx
        if len(cnt)<4: continue
        cnt=cnt.squeeze()*upscale
        cnt = cnt.tolist()
        cnt.append(cnt[0])
        cnts.append([cnt])
    p = geojson.MultiPolygon(cnts)
    feature = geojson.Feature(geometry=p)
    feature["id"]= "PathAnnotationObject"
    return [feature]

def maskfile2geojson(mask_path, mask_ext, save_dir):
    #os.makedirs(save_dir, exist_ok=True)
    mask = cv2.imread(mask_path, 0)
    #mask = post_processing(mask)
    name = mask_path.split('/')[-1].replace(mask_ext, '')
    #bimg, contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bimg, contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cnts=[]
    for cnt in contours:
        if len(cnt)<4: continue
        arclen = cv2.arcLength(cnt, True)
        epsilon = arclen * 0.0002
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cnt=approx
        if len(cnt)<4: continue
        cnt=cnt.squeeze()*4
        cnt = cnt.tolist()
        cnt.append(cnt[0])
        cnts.append([cnt])
    p = geojson.MultiPolygon(cnts)
    feature = geojson.Feature(geometry=p)
    feature["id"]= "PathAnnotationObject"
    with open(f'{save_dir}/{name}.json', 'w') as f:
        geojson.dump([feature], f, indent=4)

if __name__ == '__main__':
    mask_path = '/mnt/md0/_datasets/BCa_QiLu/svs_selected_soft_link/09518.15_M1_x10_pred.png'
    save_dir = '/mnt/md0/_datasets/BCa_QiLu/qupath_projects/adjust_gjson_postprocessing'
    mask_ext = '_M1_x10_pred.png'
    maskfile2geojson(mask_path, mask_ext,save_dir)
