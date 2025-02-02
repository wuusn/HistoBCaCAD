from abc import ABC, abstractmethod
import numpy as np
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import struct
#import kfbReader as kr
import large_image

class AbstractImage(ABC):

    def __init__(self, path, max_mag, curr_mag):
        self.path = path
        self.ext = '.'+self.path.split('.')[-1]
        self.name = path.split('/')[-1].replace(self.ext,'')
        self.maxMag = max_mag
        self.currMag = curr_mag
        self.scale = self.maxMag / self.currMag
        self.img = self.open()
        self.getSize()

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
        self.X=0
        self.Y=0
        self.x = 0
        self.y = 0
        self.patchXStride = x_stride if x_stride != None else w
        self.patchYStride = y_stride if y_stride != None else w
        self.patchW = w
        self.patchH = h if h != None else w
        return self

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.x + self.patchXStride > self.width  and self.y + self.patchYStride > self.height:
            raise StopIteration
        else:
            tmpX = self.x
            tmpY = self.y
            self.x += self.patchXStride
            if self.x > self.width:
                self.x = 0
                self.y += self.patchYStride
            roi = self.getRegion(tmpX, tmpY, self.patchW, self.patchH)
            return roi
    
    def genPatchCoords(self, psize):
        coords = []
        x=0
        y=0
        while True:
            if x+psize > self.width and y+psize > self.height:
                break
            else:
                coords.append((x,y))
                x += psize
                if x > self.width:
                    x = 0
                    y += psize
        return coords

    def genPatchCoordsAll(self):
        coords = []
        x = self.X
        y = self.Y
        x_stride = self.patchXStride
        y_stride = self.patchYStride
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
    
class Region(AbstractImage):
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
    
    def getRegion(self, x, y, w, h): # at curr mag
        x *= self.scale
        y *= self.scale
        W = w * self.scale
        H = h * self.scale
        x = int(x)
        y = int(y)
        W = int(W)
        H = int(H)
        roi = self.img.crop((x, y, x+W, y+H))
        roi = roi.resize((w, h), Image.BICUBIC)
        return np.array(roi).astype(np.uint8)


class KFB(AbstractImage):
    def __init__(self, path, max_mag, curr_mag):
        super().__init__(path, max_mag, curr_mag)

    def open(self):
        self.reader = kr.reader()
        self.reader.ReadInfo(self.path, self.currMag, False)

    def getSize(self):
        if hasattr(self, 'width') and hasattr(self, 'height'):
            return self.width, self.height
        else:
            self.width = self.reader.getWidth()
            self.height = self.reader.getHeight()
    
    def getRegion(self, x, y, w, h):
        self.reader = kr.reader()
        self.reader.ReadInfo(self.path, self.currMag, False)
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        roi =  self.reader.ReadRoi(x, y, w, h, self.currMag)
        roi = roi.copy() 
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        return np.array(roi).astype(np.uint8)


class SVS(AbstractImage):
    def __init__(self, path, max_mag, curr_mag):
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
    
    def getRegion(self, x, y, w, h):
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
        return np.array(patch).astype(np.uint8)

def WSI(path, max_mag, curr_mag):
    ext = '.'+path.split('/')[-1].split('.')[-1]
    if ext == '.kfb':
        return KFB(path, max_mag, curr_mag)
    elif ext in ['.svs', '.tiff']:
        return SVS(path, max_mag, curr_mag)
    return Region(path, max_mag, curr_mag)



# header position
headerLength = 0x64
CAP_RES = 8
BLOCK_COUNT = 16
HEIGHT = 20
WIDTH = 24
SCALE = 28
TILE_SIZE = 88
PREVIEW_POSITION = 52
LABEL_POSITION = 56
THUMB_POSITION = 60
IMAGE_POSITION = 68

IMAGE_COUNT=64
LABEL_COUNT=52
PREVIEW_COUNT=52
THUMB_COUNT=44

def readKfbThumb(path):
    f = open(path, 'rb')
    imageHeaderBytes = f.read(headerLength)

    f.seek(WIDTH, 0)
    width = struct.unpack('i', f.read(4))[0]
    f.seek(HEIGHT, 0)
    height = struct.unpack('i', f.read(4))[0]

    f.seek(THUMB_POSITION,0)
    thumbPosition = struct.unpack('l', f.read(8))[0]
    f.seek(thumbPosition,0)

    f.seek(thumbPosition+20,0)
    length = struct.unpack('i', f.read(4))[0]

    f.seek(thumbPosition+24,0)
    start = struct.unpack('l', f.read(8))[0]

    f.seek(thumbPosition+start, 0)
    thumbBytes = f.read(length)
    f.close()

    decoded = cv2.imdecode(np.frombuffer(thumbBytes, np.uint8), -1)
    thumb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    #Image.fromarray(thumb).save('/home/yxw1452/Desktop/test2.jpg')
    scale = width // thumb.shape[1]
    return thumb, scale, width, height
