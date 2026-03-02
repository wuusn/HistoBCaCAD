from turtle import width
import numpy as np
import os
import sys
import glob

np.seterr(all="ignore")
import warnings

warnings.filterwarnings('ignore')
from abc import ABC, abstractmethod
import cv2
import copy
import math
from matplotlib import cm
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
from PIL import ImageFile
# import large_image
# from histomicstk.saliency import tissue_detection

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
import time

def get_tissue_mask(low_res_img, deconvolve_first=False, n_thresholding_steps=1, sigma=0., min_size=30):
    labeled, mask = tissue_detection.get_tissue_mask(
            low_res_img, deconvolve_first=deconvolve_first,
            n_thresholding_steps=n_thresholding_steps, sigma=sigma, min_size=min_size
    )
    return mask.astype(np.uint8)

def outputs2probmap(filepath, csvpath, save_path=None):
    ts = large_image.getTileSource(filepath)
    low_res_img, _ = ts.getRegion(scale=dict(magnification=1), format=large_image.tilesource.TILE_FORMAT_NUMPY)
    mask = get_tissue_mask(low_res_img)
    cnts = mask2cnts(mask, scale=10)
    patch_size = 336
    cnt = cnts[0]
    W = cnt[2]
    H = cnt[3]
    wsi = WSI(filepath)
    wsi.X = cnt[0]
    wsi.Y = cnt[1]
    wsi.width = W
    wsi.height = H
    wsi.setIterator(patch_size)
    coords = wsi.genPatchCoordsAll()
    w = math.ceil(W/patch_size)
    h = math.ceil(H/patch_size)
    print(len(coords))
    print(w*h)

    array = np.genfromtxt(csvpath, delimiter=',')
    array = array.reshape(h,w,3)
    if save_path != None:
        with open(save_path, 'wb') as f:
            np.save(f, array)

def mask2cnts(mask, scale):
    if np.max(mask)>1 :
        mask = mask/255
        mask = mask.astype(np.uint8)

    contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        rois.append([x*scale,y*scale,w*scale,h*scale])
    return rois

class AbstractImage(ABC):

    def __init__(self, path, max_mag, curr_mag):
        self.path = path
        self.ext = '.' + self.path.split('.')[-1]
        self.name = path.split('/')[-1].replace(self.ext, '')
        self.maxMag = max_mag
        self.currMag = curr_mag
        self.scale = self.maxMag / self.currMag
        self.img = self.open()
        self.getSize()
        self.X = 0
        self.Y = 0

    def __deepcopy__(self, memodict={}):
        copyobj = type(self)(self.path, self.maxMag, self.currMag)
        copyobj.X = self.X
        copyobj.Y = self.Y
        copyobj.width = self.width
        copyobj.height = self.height
        return copyobj

    @abstractmethod
    def open(self):
        pass

    @abstractmethod
    def getSize(self):
        pass

    @abstractmethod
    def getRegion(self, x, y, w, h):
        pass

    def setIterator(self, w, h=None, x_stride=None, y_stride=None):
        self.x = self.X
        self.y = self.Y
        self.patchXStride = x_stride if x_stride != None else w
        self.patchYStride = y_stride if y_stride != None else w
        self.patchW = w
        self.patchH = h if h != None else w
        return self

    def __iter__(self):
        return self

    def __next__(self):
        #return self.iteratorSimple()
        return self.iteratorAll()

    def iteratorSimple(self):
        if self.y + self.patchH > self.Y + self.height:
            raise StopIteration
        else:
            tmpX = self.x
            tmpY = self.y
            self.x += self.patchXStride
            if self.x + self.patchW > self.X + self.width:
                self.x = self.X
                self.y += self.patchYStride
            roi = self.getRegion(tmpX, tmpY, self.patchW, self.patchH)
            return roi

    def iteratorAll(self):
        if self.y == self.Y + self.height:
            raise StopIteration
        if self.y + self.patchH > self.Y + self.height:
            self.y = self.y - (self.patchH - (self.Y + self.height - self.y))
            if self.y == self.tmpY:
                raise StopIteration

        self.tmpX = self.x
        self.tmpY = self.y
        self.x += self.patchXStride
        if self.x == self.X + self.width:
            self.x = self.X
            self.y += self.patchYStride
        elif self.x + self.patchW > self.X + self.width:
            self.x = self.x - (self.patchW - (self.X + self.width - self.x))
            if self.x == self.tmpX:
                self.x = self.X
                self.y += self.patchYStride
        patch = self.getRegion(self.tmpX, self.tmpY, self.patchW, self.patchH)
        return patch

    def genPatchCoordsSimple(self):
        coords = []
        x = self.X
        y = self.Y
        x_stride = self.patchW
        y_stride = self.patchH
        tmp_x = x
        tmp_y = y

        while True:
            if tmp_y + y_stride > y + self.height:
                break
            coords.append((tmp_x, tmp_y))
            tmp_x += x_stride
            if tmp_x + x_stride > x + self.width:
                tmp_x = x
                tmp_y += y_stride
        return coords

    def genPatchCoordsAll(self):
        coords = []
        x = self.X
        y = self.Y
        x_stride = self.patchW
        y_stride = self.patchH
        tmp_x = x
        tmp_y = y

        while True:
            if tmp_y == y + self.height:
                break
            if tmp_y + y_stride > y + self.height:
                tmp_y = tmp_y - (y_stride - (y + self.height - tmp_y))
            coords.append((tmp_x, tmp_y))
            tmp_x += x_stride
            if tmp_x == x + self.width:
                tmp_x = x
                tmp_y += y_stride
            elif tmp_x + x_stride > x + self.width:
                tmp_x = tmp_x - (x_stride - (x + self.width - tmp_x))
                if tmp_x == x:
                    tmp_x = x
                    tmp_y += y_stride
        return coords


class ROI(AbstractImage):

    def __init__(self, path, max_mag, curr_mag):
        super().__init__(path, max_mag, curr_mag)

    def open(self):
        return Image.open(self.path)

    def getSize(self):
        if hasattr(self, 'width') and hasattr(self, 'height'):
            return self.width, self.height
        else:
            w, h = self.img.size
            self.width = int(w / self.scale)
            self.height = int(h / self.scale)

    def getRegion(self, x, y, w, h):  # at curr mag
        x *= self.scale
        y *= self.scale
        W = w * self.scale
        H = h * self.scale
        x = int(x)
        y = int(y)
        W = int(W)
        H = int(H)
        roi = self.img.crop((x, y, x + W, y + H))
        roi = roi.resize((w, h), Image.BICUBIC)
        return roi
        #return np.array(roi).astype(np.uint8)

class QiLuROI(ROI):
    def __init__(self, path, max_mag=10, curr_mag=10, curr_min_size=336):
        self.curr_min_size = curr_min_size
        super().__init__(path, max_mag, curr_mag)
        
    def open(self):
        min_size = self.curr_min_size * self.scale
        min_size = int(min_size)
        img = Image.open(self.path)
        w ,h = img.size

        if w < min_size or h < min_size:
            new_h = min_size if h < min_size else h
            new_w = min_size if w < min_size else w
            pad_tran = transforms.RandomCrop((new_h,new_w), padding_mode='reflect', pad_if_needed=True)
            img = pad_tran(img)

        return img
class Patch(ROI):
    def __init__(self, path, max_mag=10, curr_mag=10):
        super().__init__(path, max_mag, curr_mag)

class QingDaoROI(QiLuROI):
    def __init__(self, path, max_mag=20, curr_mag=10):
        super().__init__(path, max_mag, curr_mag)

class AgiosPavlosROI(QiLuROI):
    def __init__(self, path, max_mag=40, curr_mag=10):
        super().__init__(path, max_mag, curr_mag)

class BCNBRoI(QiLuROI):
    def __init__(self, path, max_mag=20, curr_mag=10):
        super().__init__(path, max_mag, curr_mag)

class WSI(AbstractImage):
    def __init__(self, path, max_mag=40, curr_mag=10):
        super().__init__(path, max_mag, curr_mag)

    def open(self):
        return large_image.getTileSource(self.path)

    def getSize(self):
        if hasattr(self, 'width') and hasattr(self, 'height'):
            return self.width, self.height
        else:
            w = self.img.sizeX
            h = self.img.sizeY
            self.width = int(w / self.scale)
            self.height = int(h / self.scale)

    def getRegion(self, x, y, w, h):  # at curr mag
        x *= self.scale
        y *= self.scale
        W = w * self.scale
        H = h * self.scale
        x = int(x)
        y = int(y)
        W = int(W)
        H = int(H)
        patch, _ = self.img.getRegion(
                region= dict(left=x, top=y, width=W, height=H), # left: distance to left, top: distance to top
                format = large_image.tilesource.TILE_FORMAT_PIL
        )
        patch = patch.convert(mode='RGB')
        patch = patch.resize((w, h), Image.BICUBIC)
        return patch
        #return np.array(roi).astype(np.uint8)

class ProbabilityMap(AbstractImage):
    def __init__(self, path, max_mag=1, curr_mag=1):
        super().__init__(path, max_mag, curr_mag)

    def open(self):
        with open(self.path, 'rb') as f:
            img = np.load(f)
        return img

    def getSize(self):
        if hasattr(self, 'width') and hasattr(self, 'height'):
            return self.width, self.height
        else:
            self.width = self.img.shape[1]
            self.height = self.img.shape[0]

    def getRegion(self, x, y, w, h):
        roi = self.img[y:y+h, x:x+w, :]
        return roi

    def setIterator(self, w, h=None, x_stride=None, y_stride=None):
        self.x = self.X
        self.y = self.Y
        self.patchXStride = x_stride if x_stride != None else w
        self.patchYStride = y_stride if y_stride != None else w
        self.patchW = w
        self.patchH = h if h != None else w

        if self.patchW > self.width:
            self.patchW = self.width

        if self.patchH > self.height:
            self.patchH = self.height

        return self
