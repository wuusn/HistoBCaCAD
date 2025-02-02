from data.wsi import *
from data.wsi2patch import wsi2PatchFromMaskOnInOut
import PIL
from PIL import Image
import os
from skimage.morphology import remove_small_objects, binary_opening, disk
from skimage import io, color, img_as_ubyte
import skimage
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import random
import sklearn
import sklearn.model_selection

def post_processing(mask, scale=1):
    mask= mask / 255 if np.max(mask)>1 else mask
    mask = mask.astype(bool)
    area_thresh = 200//scale
    mask_opened = remove_small_objects(mask, min_size=area_thresh)
    mask_removed_area = ~mask_opened & mask
    mask = mask_opened > 0

    min_size = 300//scale
    img_reduced = skimage.morphology.remove_small_holes(mask, area_threshold=min_size)
    img_small = img_reduced & np.invert(mask)
    mask = img_reduced
    mask = mask.astype(np.uint8)
    kernel = np.ones((5,5), dtype=np.uint8)
    new = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return new


def localtest():
    #kfb_path = '/raid10/_datasets/Breast/QiLu/NonTumor/201943541.kfb'
    #mask_path = '/raid10/_datasets/Breast/QiLu/kfb_normal_results/201943541_pred.png'
    kfb_path = '/raid10/_datasets/Oral/TMA/OSU/oral_cavit1_11-C.jpg'
    mask_path = '/raid10/_datasets/Oral/TMA/OSU/oral_cavit1_11-C.png'
    target_folder = '/home/yxw1452/Desktop/tmp3'
    max_mag = 40
    curr_mag = 10
    mask_mag = 40
    scale = curr_mag // mask_mag
    scale = int(scale)
    scale = 1 if scale == 0 else scale
    psize = 256#512
    interp_method=PIL.Image.BICUBIC
    #wsi = KFB(kfb_path, max_mag, curr_mag)
    wsi = Region(kfb_path, max_mag, curr_mag)
    nl = cv2.imread(mask_path, 0)
    #post = post_processing(nl)
    nl2 = nl/255 if np.max(nl)>1 else nl
    nl2 = nl2.astype(np.uint8)
    edge = binary_dilation(nl2==1, iterations=1) & ~nl2
    ys,xs = (edge > 0).nonzero() #
    #print(xs[0], ys[0])
    size = len(xs)
    idxs = list(range(0, size))
    sel = idxs
    sel_size = size//3
    #_, sel = sklearn.model_selection.train_test_split(idxs, test_size=sel_size, random_state=56)
    for i in sel:
        x = xs[i]*scale - psize//2#64
        y = ys[i] - #64
        if x <0 or y <0:
            continue
        region_mask = nl[y:y+128, x:x+128]
        x = x*scale
        y = y*scale
        #Image.fromarray(region_mask).save(f'/home/yxw1452/Desktop/tmp2/{x}_{y}_mask.png')
        #Image.fromarray(wsi.getRegion(x,y,512,512)).save(f'/home/yxw1452/Desktop/tmp2/{x}_{y}_roi.png')

    # In, out = in
    #stride = psize // scale // 2 //2 
    stride = 2
    #post = post_processing(nl2)
    out = binary_erosion(nl2==1, iterations=stride)
    #out = ~binary_dilation(nl2==1, iterations=stride)
    out = out.astype(np.uint8)
    #out = cv2.cv.fromarray(out)
    tiny_out = cv2.resize(out, (out.shape[1]//(psize//scale),out.shape[0]//(psize//scale)), interp_method)
    ys,xs = (tiny_out > 0).nonzero() #
    size = len(xs)
    print(size)
    idxs = list(range(0, size))
    if size < sel_size:
        sel = list(range(0, size))
    else:
        _, sel = sklearn.model_selection.train_test_split(idxs, test_size=sel_size, random_state=56)
    for i in sel:
        x = xs[i]*psize
        y = ys[i]*psize
        if x < 0 or y <0:
            continue
        np_patch = wsi.getRegion(x,y,psize,psize)
        patch = Image.fromarray(np_patch)
        patch.save(f'{target_folder}/{name}_{i}_{int(x)}_{int(y)}_in.png')

if __name__ == '__main__':
    path = '/raid10/_datasets/Breast/QiLu/NonTumor/201943541.kfb'
    tar_dir = '/home/yxw1452/Desktop/tmp3'
    params = dict(src_dir='', max_mag=40, curr_mag=10, mask_mag=2.5, mask_dir = '/raid10/_datasets/Breast/QiLu/kfb_normal_results', mask_ext='_pred.png',size=512)
    #wsi2PatchFromMaskOnInOut(path, tar_dir, '.png', params)
    localtest()
