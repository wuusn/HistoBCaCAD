from cypath.data.wsi import WSI
from cypath.data.multiRun import multiRunStarmap
from PIL import Image
import numpy as np
import os
import glob

def mergeOnePatch(patch_path, save_scale, split, idx1, idx2):
    name = os.path.basename(patch_path)
    ext = '.'+name.split('.')[-1]
    name = name.replace(ext, '')
    im = Image.open(patch_path)
    w,h = im.size
    w = w//save_scale
    h = h//save_scale
    a = np.array(im)
    if len(a.shape) > 2:
        im = im.resize((w,h))
        im = np.array(im).astype(np.uint8)
    else:
        im = im.convert('1')
        im = im.resize((w,h))
        im = np.array(im).astype(np.uint8)*255
    n = im.ndim
    splits = name.split(split)
    x = splits[idx1]
    y = splits[idx2]
    x = int(x)//save_scale
    y = int(y)//save_scale

    return x,y,im

def mergePatch(name, W,H,N, psize, patch_dir, save_dir, patch_ext, save_ext, save_scale, split, idx1, idx2):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    scale = save_scale
    patch_paths = glob.glob(f'{patch_dir}/*{patch_ext}')
    if N==1:
        M = np.zeros((H//scale+psize,W//scale+psize))
    else:
        M = np.zeros((H//scale+psize,W//scale+psize,N))

    count = np.zeros_like(M)
    res = multiRunStarmap(mergeOnePatch, patch_paths, save_scale, split, idx1, idx2)
    if N==1:
        for re in res:
            x,y,im = re
            #print(M[y:y+psize//scale, x:x+psize//scale].shape, im.shape)
            M[y:y+psize//scale, x:x+psize//scale] += im
            count[y:y+psize//scale, x:x+psize//scale] += 1
        M = M[:H//scale,:W//scale]
        count = count[:H//scale,:W//scale]
        M = np.divide(M, count)
        #M = M == 255
        #M = M.astype(np.uint8)
        #M = M * 255
        M = M.astype(np.uint8)
    else:
        for re in res:
            x,y,im = re
            #print(H//scale,W//scale,y,x)
            M[y:y+psize//scale, x:x+psize//scale,:] += im
            count[y:y+psize//scale, x:x+psize//scale, :] += 1
        M = M[:H//scale,:W//scale,:]
        count = count[:H//scale,:W//scale]
        M = np.divide(M, count)
    M = M.astype(np.uint8)
    if save_ext and save_dir:
        im = Image.fromarray(M)
        im.save(f'{save_dir}/{name}{save_ext}')
    return M


if __name__ == '__main__':
    # test
    import time
    start = time.time()

    name = '00026.18'
    path = '/raid10/_datasets/Breast/QiLu/Tumor/00026.18.kfb'
    wsi = WSI(path, 40, 10)
    W,H = wsi.getSize()
    N=3
    psize = 512
    patch_dir = '/tmp/test_wsi2patch'
    save_ext = '.png'
    patch_ext = '.jpg'
    save_dir = '/tmp'
    mergePatch(name,W,H,N,psize,patch_dir,save_dir, patch_ext, save_ext, 4, '-', 3,4)
     
    end = time.time()
    print('time', (end-start)/60)
    # 0.13 min
