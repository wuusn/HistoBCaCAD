import cv2
import numpy as np

def getBoxFromAnno(np_gray):
    x,y,w,h = cv2.boundingRect(np_gray)
    box = (x,y,w,h)
    return box

def getBoxFromAnnoPath(anno_path):
    anno = cv2.imread(anno_path, 0)
    return getBoxFromAnno(anno)

def getRegionOfBox(box, anno):
    assert anno.ndim == 2
    x,y,w,h = box
    roi = anno[y:y+h, x:x+w]
    return roi

def getBoxRegionFromAnnoPath(anno_path):
    anno = cv2.imread(anno_path, 0)
    box = getBoxFromAnno(anno)
    roi = getRegionOfBox(box, anno)
    return roi

if __name__ == '__main__':
    #anno_path = '/mnt/md0/_datasets/OralCavity/WSI/SFVA/Masks/epi_unet_nonexpert/SP06-1112 D3_tuned.png'
    #anno_path = '/mnt/md0/_datasets/OralCavity/WSI/SFVA/Masks/epi_unet_nonexpert/SP06-1112 D5_tuned.png' 
    anno_path = '/mnt/md0/_datasets/OralCavity/WSI/SFVA/Masks/epi_unet_nonexpert/SP06-2068 A1_tuned.png' 
    roi = getBoxRegionFromAnnoPath(anno_path)
    cv2.imwrite('/tmp/test.png', roi)
