import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
import glob
import os

def compareMasks(m1, m2, threshold=.5): # m1 is pred, m2 is groud truth
    assert m1.shape == m2.shape
    t = threshold * 255
    l1 = (m1>t).astype(np.uint8) if np.max(m1) > 1 else m1>=threshold
    l2 = (m2/255).astype(np.uint8) if np.max(m2) > 1 else m2
    l1 = l1.flatten()
    l2 = l2.flatten()
    metric = getScores(l1, l2)
    #m1 = m1 / 255
    #m2 = m2 / 255
    return metric#, (m2, m1)

def maskDifferFP(gt, p):
    assert gt.shape == p.shape
    gt = (gt==255).astype(np.uint8) if np.max(gt) > 1 else gt
    p = (p==255).astype(np.uint8) if np.max(p) > 1 else p
    res = (gt==0) & (p==1)
    res = res.astype(np.uint8)
    return res

def maskDifferFN(gt, p):
    assert gt.shape == p.shape
    gt = (gt==255).astype(np.uint8) if np.max(gt) > 1 else gt
    p = (p==255).astype(np.uint8) if np.max(p) > 1 else p
    res = (gt==1) & (p==0)
    res = res.astype(np.uint8)
    return res

def excludeMask(m, e):
    assert m.shape == e.shape
    e = e.astype(np.bool)
    m[e] = 0
    return m

def getScores(a, b): # a is pred, b is ground truth
    e = 1e-6
    res = confusion_matrix(b, a, labels=[0,1]).ravel() # set labels always has 4 values to pack
    tn, fp, fn, tp = res
    acc = (tp+tn) / (tp+tn+fp+fn+e)
    precision = tp / (tp+fp+e)
    recall = tp / (tp+fn+e)
    specificity = tn / (tn+fp+e)
    f1 = 2*tp/(2*tp+fp+fn+e)
    metric=dict(acc=acc, precision=precision, recall=recall, specificity=specificity, f1=f1)
    return metric

def compareImageFileShape(a_path, b_path):
    try:
        a = cv2.imread(a_path, 0)
        b = cv2.imread(b_path, 0)
        if a.shape != b.shape:
            return False
        return True
    except:
        return False

def compareImageDirShape(a_dir, b_dir, a_ext, b_ext):
    a_paths = glob.glob(f'{a_dir}/*{a_ext}')
    b_paths = glob.glob(f'{b_dir}/*{b_ext}')
    if len(a_paths) != len(b_paths):
        print('Error: dir size not compare!', len(a_paths), len(b_paths))
    if len(a_paths) > len(b_paths):
        for a_path in a_paths:
            name = a_path.split('/')[-1].replace(a_ext, '')
            b_path = f'{b_dir}/{name}{b_ext}'
            if os.path.exists(b_path)==False:
                print(f'Error: {b_path} not exists')
            if compareImageFileShape(a_path, b_path)==False:
                print(f'Error: {name} not compare!')
    else:
        for b_path in b_paths:
            name = b_path.split('/')[-1].replace(b_ext, '')
            a_path = f'{a_dir}/{name}{a_ext}'
            if os.path.exists(a_path)==False:
                print(f'Error: {a_path} not exists')
            if compareImageFileShape(b_path, a_path)==False:
                print(f'Error: {name} not compare!')

def getRatioOfMaskPaths(src_dir, src_ext):
    mask_paths = glob.glob(f'{src_dir}/*{src_ext}')
    a = [0, 0]
    min_shape = (10000,10000)
    max_shape = (0,0)
    width = 0
    height = 0
    for mask_path in mask_paths:
        mask = cv2.imread(mask_path, 0)
        mask = (mask == 255).astype(np.uint8) if np.max(mask) > 1 else mask
        n_pixels = mask.shape[0] * mask.shape[1]
        one = np.sum(mask)
        a[0]+=one
        a[1]+=n_pixels
        if n_pixels < min_shape[0] * min_shape[1]:
            min_shape = mask.shape
        if n_pixels > max_shape[0] * max_shape[1]:
            max_shape = mask.shape
        width += mask.shape[1]
        height += mask.shape[0]
    print(len(mask_paths))
    print(a[0]/(a[1]-a[0]))
    print(min_shape)
    print(max_shape)
    print(width/len(mask_paths), height/len(mask_paths))
        

if __name__ == '__main__':
    #a = np.ones((255,255,3)).astype(np.uint8)
    #b = np.zeros((255,255,3)).astype(np.uint8)
    #b[1,1,1]=1
    #print(compareMasks(a,b))
    #a = np.ones((2,6,6))
    #b = np.ones((2,6,6))*255
    #a = excludeMask(a,b)
    #print(a)
    #a_dir = '/mnt/md0/_datasets/OralCavity/WSI/UCSF/Masks/epi_unet_nonexpert'
    #b_dir = '/mnt/D/Oral/train_wsi_OnInOut/VUMC/save/base_unet/2.5x_full/ucsf'
    #compareImageDirShape(a_dir, b_dir, '_tuned.png', '_pred.png')
    getRatioOfMaskPaths('/home/yxw1452/data/trials/breast/qilu/10xRegionFromAnno20210219/*', '_mask.png')
