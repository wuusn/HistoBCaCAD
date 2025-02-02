from .wsi import *
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
random.seed(56)

#def wsi2GridPatch(path, tar_dir,save_ext,  mag, size, ratio):
def wsi2GridPatch(path, tar_dir, tar_ext, param):
    save_ext = tar_ext
    dir_name = os.path.dirname(path)
    src_dir = param.get('src_dir')
    max_mag = param.get('max_mag', 10)
    curr_mag = param.get('curr_mag', 10)
    size = param.get('size', [256, 256]) 
    ratio = param.get('ratio', 1.0)
    groupByName = param.get('groupByName', False)
    ext = path.split('.')[-1]
    name = path.split('/')[-1].replace(f'{ext}', '')
    if ext == 'kfb':
        wsi = KFB(path, max_mag, curr_mag)
        #print(name, wsi.getSize())
    else:
        wsi = Region(path, max_mag, curr_mag)
    w,h = size
    x_stride = w * ratio
    y_stride = h * ratio
    wsi.setIterator(w, h, x_stride, y_stride)
    i = 0
    if groupByName:
        save_dir = dir_name.replace(src_dir, tar_dir)
        save_dir = f'{save_dir}/{name}'
    else:
        save_dir = dir_name.replace(src_dir, tar_dir)
    os.makedirs(save_dir, exist_ok=True)
    for np_im in wsi:
        im = Image.fromarray(np_im)
        im.save(f'{save_dir}/{name}_{i}{save_ext}')
        i += 1

def post_processing(mask, scale=1):
    mask= mask / 255 if np.max(mask)>1 else mask
    mask = mask.astype(np.bool)
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

def wsi2PatchRandom(path, tar_dir, tar_ext, param):
    dir_name = os.path.dirname(path)
    src_dir = param.get('src_dir')
    max_mag = param.get('max_mag')
    max_mag = int(max_mag) if max_mag != None else 10
    curr_mag = param.get('curr_mag')
    curr_mag = int(curr_mag) if curr_mag != None else 10
    mask_mag = param.get('mask_mag')
    mask_mag = float(mask_mag) if mask_mag != None else 10
    groupByName = True if param.get('groupByName')==True else False
    saveWithMask = True if param.get('saveWithMask')==True else False
    ratio = param.get('ratio')
    ratio = int(ratio) if ratio is not None else 1
    ext = path.split('.')[-1]
    name = path.split('/')[-1].replace(f'.{ext}', '')
    mask_dir = param.get('mask_dir')
    mask_ext = param.get('mask_ext')
    mask_path = f'{mask_dir}/{name}{mask_ext}'
    scale = curr_mag / mask_mag
    #scale = int(scale)
    size = param.get('size') 
    psize = int(size)
    interp_method=PIL.Image.BICUBIC
    if groupByName:
        target_folder = f'{tar_dir}/{name}'
    else:
        target_folder = tar_dir
    os.makedirs(target_folder, exist_ok=True)
    if ext == 'kfb':
        wsi = KFB(path, max_mag, curr_mag)
    else:
        wsi = Region(path, max_mag, curr_mag)
    mask = Region(mask_path, mask_mag, curr_mag)

    w, h = wsi.getSize()
    mount = w//psize * h//psize * ratio
    i = 0
    for i in range(mount):
        x = random.randint(0, w)
        y = random.randint(0, h)
        np_patch = wsi.getRegion(x,y,psize,psize)
        patch = Image.fromarray(np_patch)
        patch.save(f'{target_folder}/{name}_{i}_{int(x)}_{int(y)}_on{tar_ext}')

        if saveWithMask:
            np_mask_patch= mask.getRegion(x,y,psize,psize)
            mask_patch = Image.fromarray(np_mask_patch)
            mask_patch.save(f'{target_folder}/{name}_{i}_{int(x)}_{int(y)}_on{mask_ext}')

    
def wsi2PatchRandomInMask(path, tar_dir, tar_ext, param):
    dir_name = os.path.dirname(path)
    src_dir = param.get('src_dir')
    src_ext = param.get('src_ext')
    max_mag = param.get('max_mag')
    max_mag = int(max_mag) if max_mag != None else 10
    curr_mag = param.get('curr_mag')
    curr_mag = int(curr_mag) if curr_mag != None else 10
    mask_mag = param.get('mask_mag')
    mask_mag = float(mask_mag) if mask_mag != None else 10
    groupByName = True if param.get('groupByName')==True else False
    saveWithMask = True if param.get('saveWithMask')==True else False
    ratio = param.get('ratio')
    ratio = int(ratio) if ratio is not None else 1
    ext = src_ext
    name = path.split('/')[-1].replace(f'{ext}', '')
    mask_dir = param.get('mask_dir')
    mask_ext = param.get('mask_ext')
    mask_path = f'{mask_dir}/{name}{mask_ext}'
    if curr_mag > mask_mag:
        scale = curr_mag / mask_mag
    else:
        scale = mask_mag / curr_mag
    #scale = int(scale)
    size = param.get('size') 
    psize = int(size)
    interp_method=PIL.Image.BICUBIC
    if groupByName:
        target_folder = f'{tar_dir}/{name}'
    else:
        target_folder = tar_dir
    os.makedirs(target_folder, exist_ok=True)
    if ext == '.kfb':
        wsi = KFB(path, max_mag, curr_mag)
    else:
        wsi = Region(path, max_mag, curr_mag)
    mask = Region(mask_path, mask_mag, curr_mag)
    nl = cv2.imread(mask_path, 0)

    w, h = wsi.getSize()
    mount = w//psize * h//psize * ratio
    #post = post_processing(nl)
    nl2 = nl/255 if np.max(nl)>1 else nl
    nl2 = nl2.astype(np.uint8)
    ys,xs = (nl2 > 0).nonzero() #
    size = len(xs)
    sel_size = int(size//(psize))
    i = 0
    for i in range(mount):
        idx = random.randint(0, size-1)
        x = xs[idx] // scale - psize // 2
        y = ys[idx] // scale - psize // 2
        np_patch = wsi.getRegion(x,y,psize,psize)
        patch = Image.fromarray(np_patch)
        patch.save(f'{target_folder}/{name}-{int(x)}_{int(y)}{tar_ext}')

        if saveWithMask:
            np_mask_patch= mask.getRegion(x,y,psize,psize)
            mask_patch = Image.fromarray(np_mask_patch)
            mask_patch.save(f'{target_folder}/{name}-{int(x)}_{int(y)}{mask_ext}')

def wsi2PatchFromMaskOnInOut(path, tar_dir, tar_ext, param):
    dir_name = os.path.dirname(path)
    src_dir = param.get('src_dir')
    max_mag = param.get('max_mag', 10)
    curr_mag = param.get('curr_mag', 10)
    mask_mag = param.get('mask_mag')
    groupByName = param.get('groupByName', False)
    saveWithMask = param.get('saveWithMask', False)
    ratio = param.get('ratio', 1)
    ext = path.split('.')[-1]
    name = path.split('/')[-1].replace(f'.{ext}', '')
    mask_dir = param.get('mask_dir')
    mask_ext = param.get('mask_ext')
    mask_path = f'{mask_dir}/{name}{mask_ext}'
    scale = curr_mag / mask_mag
    #scale = int(scale)
    size = param.get('size') 
    psize = int(size)
    interp_method=PIL.Image.BICUBIC
    if ext == 'kfb':
        wsi = KFB(path, max_mag, curr_mag)
    else:
        wsi = Region(path, max_mag, curr_mag)

    nl = cv2.imread(mask_path, 0)
    mask = Region(mask_path, mask_mag, curr_mag)
    #post = post_processing(nl)
    nl2 = nl/255 if np.max(nl)>1 else nl
    nl2 = nl2.astype(np.uint8)

    if groupByName:
        target_folder = f'{tar_dir}/{name}'
    else:
        target_folder = tar_dir
    os.makedirs(target_folder, exist_ok=True)

    # On
    edge = binary_dilation(nl2==1, iterations=1) & ~nl2
    ys,xs = (edge > 0).nonzero() #
    size = len(xs)
    idxs = list(range(0, size))
    #sel_size = int(size//(psize//scale)//3)
    sel_size = int(size//(psize))*ratio
    #print(sel_size, size)
    #return
    if size < sel_size:
        sel = list(range(0, size))
    else:
        _, sel = sklearn.model_selection.train_test_split(idxs, test_size=sel_size, random_state=56)
    for i in sel:
        x = xs[i]*scale-psize//2
        y = ys[i]*scale-psize//2
        x = int(x) - psize//2
        y = int(y) - psize//2
        if x < 0 or y <0:
            continue
        np_patch = wsi.getRegion(x,y,psize,psize)
        patch = Image.fromarray(np_patch)
        patch.save(f'{target_folder}/{name}_{i}_{int(x)}_{int(y)}_on{tar_ext}')

        if saveWithMask:
            np_mask_patch= mask.getRegion(x,y,psize,psize)
            mask_patch = Image.fromarray(np_mask_patch)
            mask_patch.save(f'{target_folder}/{name}_{i}_{int(x)}_{int(y)}_on.png')

    # Out 
    stride = psize//scale//2
    stride = int(stride)
    out = ~binary_dilation(nl2==1, iterations=stride) #& ~ nl2 #
    out = out.astype(np.uint8)
    tiny_out = cv2.resize(out, (out.shape[1]//int(psize//scale),out.shape[0]//int(psize//scale)), interp_method)
    ys,xs = (tiny_out > 0).nonzero() #
    size = len(xs)
    idxs = list(range(0, size))

    if size == 0:
        sel = []
    elif size <= sel_size:# or size <= 2:
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
        patch.save(f'{target_folder}/{name}_{i}_{int(x)}_{int(y)}_out{tar_ext}')
        if saveWithMask:
            np_mask_patch= mask.getRegion(x,y,psize,psize)
            mask_patch = Image.fromarray(np_mask_patch)
            mask_patch.save(f'{target_folder}/{name}_{i}_{int(x)}_{int(y)}_out.png')

    # In
    #out = ~binary_dilation(nl2==1, iterations=stride) & nl2 #
    stride = 2
    inside = binary_erosion(nl2==1, iterations=stride)
    inside = inside.astype(np.uint8)
    #print(name, np.sum(inside))
    tiny_inside = cv2.resize(inside, (inside.shape[1]//int(psize//scale),inside.shape[0]//int(psize//scale)), interp_method)
    ys,xs = (tiny_inside > 0).nonzero() #
    size = len(xs)
    idxs = list(range(0, size))
    if size == 0:
        sel = []
    elif size <= sel_size:# or size <= 2:
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
        patch.save(f'{target_folder}/{name}_{i}_{int(x)}_{int(y)}_in{tar_ext}')
        if saveWithMask:
            np_mask_patch= mask.getRegion(x,y,psize,psize)
            mask_patch = Image.fromarray(np_mask_patch)
            mask_patch.save(f'{target_folder}/{name}_{i}_{int(x)}_{int(y)}_in.png')
