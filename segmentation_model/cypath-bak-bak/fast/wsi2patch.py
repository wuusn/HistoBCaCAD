from cypath.data.wsi import WSI
from cypath.data.multiRun import multiRunStarmapN, multiRunStarmap
from PIL import Image
import os

def wsi2OnePatch(path, max_mag, curr_mag, coord, psize, save_dir, save_ext):
    x,y = coord
    wsi = WSI(path, max_mag, curr_mag)
    roi = wsi.getRegion(x,y,psize,psize)
    im = Image.fromarray(roi)
    im.save(f'{save_dir}/{wsi.name}-{wsi.maxMag}-{wsi.currMag}-{x}-{y}{save_ext}')

def wsi2Patch(path, save_dir, save_ext, max_mag, curr_mag, psize):
    os.makedirs(save_dir, exist_ok=True)
    wsi = WSI(path, max_mag, curr_mag)
    wsi.setIterator(psize,psize, psize, psize)
    #wsi.setIterator(psize,psize, psize*1//2, psize*1//2)
    w,h = wsi.getSize()
    coords = wsi.genPatchCoordsAll()
    #coords = wsi.genPatchCoords(psize)
    #multiRunStarmapN(20, wsi2OnePatch, path, coords, psize, save_dir, save_ext)
    multiRunStarmap(wsi2OnePatch, path, max_mag, curr_mag, coords, psize, save_dir, save_ext)

if __name__ == '__main__':
    # test
    import time
    start = time.time()

    path = '/raid10/_datasets/Breast/QiLu/Tumor/00026.18.kfb'
    max_mag = 40
    curr_mag = 10
    psize = 512
    save_dir = '/tmp/test_wsi2patch'
    save_ext = '.jpg'
    wsi2Patch(path, save_dir, save_ext, max_mag, curr_mag, psize)
     
    end = time.time()
    print('time', (end-start)/60)
    # 0.21 min
