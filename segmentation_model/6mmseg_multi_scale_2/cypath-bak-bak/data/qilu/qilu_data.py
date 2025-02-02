#from . import kfbReader as kr
import kfbReader as kr
import xml.etree.ElementTree as ET
import cv2
import copy
import numpy as np
from skimage.draw import polygon2mask
import os

class QiluWSI():
    def __init__(self, kfb_path, xml_path=None, curr_mag=10, max_mag=40):
        self.name = kfb_path.split('/')[-1].replace('.kfb', '')
        if os.path.exists(kfb_path)==False:
            raise Exception(f'{self.name} file not exists')
        self.kfb_path = kfb_path
        self.max_mag = max_mag
        self.xml_path = xml_path
        self.rectangles = None
        self.whole_anno = None
        self.reader = kr.reader()
        self.curr_mag = curr_mag
        self.scale = int(max_mag//curr_mag)
        self.reader.ReadInfo(self.kfb_path, curr_mag, False)
        if self.xml_path != None:
            self.set_rectangles()
            #self.check_rects_overlap()
            #self.set_children()
            #self.check_rects_no_child()
            #self.norm_children()
            self.get_whole_anno_without_rect(self.scale)

    def size(self, mag):
        return self.reader.getWidth(), self.reader.getHeight()

    def read_roi_local(self, x, y, w, h): #x,y in current mag
        roi = self.reader.ReadRoi(x,y,w,h,self.curr_mag) #numpy #local x,y
        return roi.copy() ## copy is critical, to avoid memory error, Segmentation fault (core dumped)

    def read_roi(self, x, y, w, h, scale): #global x,y
        #scale = self.max_mag // self.curr_mag
        x=x//scale
        y=y//scale
        roi = self.reader.ReadRoi(x,y,w,h,self.curr_mag) #numpy #local x,y
        return roi.copy() ## copy is critical, to avoid memory error, Segmentation fault (core dumped)

    def save_roi(self, x, y, w, h, scale, tar_path):
        roi = self.read_roi_global(x,y,w,h, scale)
        cv2.imwrite(tar_path, roi)

    def read_rects(self, scale):
        rois = {}
        for rect in self.rectangles:
            rect_name = rect.name.split('_')[-1]
            minx = min([c[0] for c in rect.coords])
            maxx = max([c[0] for c in rect.coords])
            miny = min([c[1] for c in rect.coords])
            maxy = max([c[1] for c in rect.coords])
            w = maxx-minx
            h = maxy-miny
            roi = self.read_roi(minx,miny,w//scale,h//scale,scale)
            rois[rect_name] = roi
        return rois

    def save_rects(self, scale, tar_dir):
        rois = self.read_rects(scale)
        for rect_name, roi in rois.items():
            cv2.imwrite(f'{tar_dir}/{self.name}_{rect_name}.png', roi)

    def anno2mask(self, rect, scale):
        minx = min([c[0] for c in rect.coords])
        maxx = max([c[0] for c in rect.coords])
        miny = min([c[1] for c in rect.coords])
        maxy = max([c[1] for c in rect.coords])
        w = maxx-minx
        h = maxy-miny
        image_shape = (h//scale, w//scale)
        mask = np.zeros(image_shape).astype(np.uint8)
        for anno in rect.children:
            relative_coords=[]
            for x,y in anno.coords:
                x-=minx
                y-=miny
                relative_coords.append([y//scale, x//scale])
            sub_mask = polygon2mask(image_shape, relative_coords)
            sub_mask.dtype = np.uint8
            #cv2.imwrite(f'/tmp/Qilu/201906716/{self.name}_{anno.name}.png', sub_mask*255)
            mask = mask | sub_mask
        return mask

    def get_region_on_whole_mask(self, minx, maxx, miny, maxy):
        if self.whole_anno is None:
            self.get_whole_anno_without_rect(self.scale)
        return self.whole_anno[miny:maxy, minx:maxx]

    def save_rect_mask(self, tar_dir):
        for rect in self.rectangles:
            minx = min([c[0] for c in rect.coords])
            maxx = max([c[0] for c in rect.coords])
            miny = min([c[1] for c in rect.coords])
            maxy = max([c[1] for c in rect.coords])
            x = minx
            y = miny
            w = maxx - minx
            h = maxy - miny
            mask = self.get_region_on_whole_mask(minx//self.scale, maxx//self.scale, miny//self.scale, maxy//self.scale)
            tumor_type = rect.tumor_type
            tumor_grade = rect.tumor_grade

            roi = self.read_roi(x,y,w//self.scale,h//self.scale,self.scale)
            rect_name = rect.name.split('_')[-1]
            #save_dir = f'{tar_dir}/{self.name}/{tumor_type}-{tumor_grade}'
            save_dir = f'{tar_dir}/{self.name}'
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(f'{save_dir}/{self.name}-{rect_name}-{tumor_type}-{tumor_grade}.png', mask*255)
            cv2.imwrite(f'{save_dir}/{self.name}-{rect_name}-{tumor_type}-{tumor_grade}.jpg', roi)


    def get_whole_anno_without_rect(self, scale):
        if self.whole_anno is not None:
            return self.whole_anno
        annos = self.xml2qilu_anno(shape='closedCurve')
        relative_coords = []
        w,h = self.size(self.max_mag)
        image_shape = (h,w)
        mask = np.zeros(image_shape).astype(np.uint8)
        for anno in annos:
            relative_coords=[]
            for x,y in anno.coords:
                relative_coords.append([y//scale, x//scale])
            sub_mask = polygon2mask(image_shape, relative_coords)
            sub_mask.dtype = np.uint8
            mask = mask | sub_mask
        #cv2.imwrite(f'/tmp/{self.name}.png', mask*255)
        self.whole_anno =  mask
        return self.whole_anno

    def save_annos(self, scale, tar_dir):
        for rect in self.rectangles:
            rect_name = rect.name.split('_')[-1]
            mask = self.anno2mask(rect, scale)
            cv2.imwrite(f'{tar_dir}/{self.name}_{rect_name}_mask.png', mask*255)

    def save_pair(self, scale, tar_dir):
        rois = self.read_rects(scale)
        for rect in self.rectangles:
            rect_name = rect.name.split('_')[-1]
            mask = self.anno2mask(rect, scale)
            roi = rois[rect_name]
            if mask.shape[0] < 50:
                continue
            #bg = np.any(roi<=[220,220,220], axis=-1)
            #bg = bg.astype(np.uint8)
            #mask = mask&bg
            cv2.imwrite(f'{tar_dir}/{self.name}_{rect_name}.png', roi)
            cv2.imwrite(f'{tar_dir}/{self.name}_{rect_name}_mask.png', mask*255)

    def xml2list(self):
        if self.xml_path == None:
            raise Exception(f'[{self.name}] No XML!')
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        annos = []
        for anno in root.iter('Annotation'):
            anno.attrib['Coordinates']=[]
            for coord in anno.iter('Coordinate'):
                anno.attrib['Coordinates'].append(coord.attrib)
            annos.append(anno.attrib)
        return annos

    def xml2qilu_anno(self, shape=None):
        annos = self.xml2list()
        shapes = {}
        qilu_annos = []
        for anno in annos:
            qilu_anno = QiluAnno.from_dict(anno)
            if shapes.get(qilu_anno.shape)==None:
                shapes[f'{qilu_anno.shape}'] = 1
            else:
                shapes[f'{qilu_anno.shape}'] += 1
            if shape==None:
                qilu_annos.append(qilu_anno)

            elif shape==qilu_anno.shape:
                minx = min([c[0] for c in qilu_anno.coords])
                maxx = max([c[0] for c in qilu_anno.coords])
                miny = min([c[1] for c in qilu_anno.coords])
                maxy = max([c[1] for c in qilu_anno.coords])
                if (maxx-minx)*(maxy-miny) < 100:
                    continue
                else:
                    qilu_annos.append(qilu_anno)
        #print(shapes)
        return qilu_annos

    def set_rectangles(self):
        annos = self.xml2qilu_anno(shape='rectangle')
        self.rectangles = annos

    def set_children(self):
        #mistake = 10
        annos = [anno for anno in self.xml2qilu_anno() if anno.shape!='rectangle']
        for anno in annos:
            parent = QiluAnno.find_max_relation(self.rectangles, anno)
            if parent!=None:
                QiluAnno.set_relation(parent, anno)
            else:
                pass
                #if len(anno.coords) > mistake:
                #    raise Exception(f'Wrong Anno [{self.name}]: child [{anno.name}] has no parent')

    def check_rects_overlap(self):
        rects = self.rectangles
        for i in range(len(rects)):
            r1 = rects[i]
            for j in range(i+1, len(rects)):
                r2 = rects[j]
                if self.check_rect_overlap(r1,r2):
                    raise Exception(f'Wrong Anno [{self.name}]: Rectangle Overlap!')

    def check_rects_no_child(self):
        for rect in self.rectangles:
            if rect.children == None:
                raise Exception(f'Wrong Anno [{self.name}]Empty Rectangle!')


    def norm_children(self):
        for rect in self.rectangles:
            minx = min([c[0] for c in rect.coords])
            maxx = max([c[0] for c in rect.coords])
            miny = min([c[1] for c in rect.coords])
            maxy = max([c[1] for c in rect.coords])
            for anno in rect.children:
                coords = anno.coords
                for i in range(len(coords)):
                    x,y=coords[i]
                    if x<minx:
                        x=minx
                    if x>maxx:
                        x=maxx
                    if y<miny:
                        y=miny
                    if y>maxy:
                        y=maxy
                    coords[i] = (x,y)

    @staticmethod
    def check_rect_overlap(r1, r2):
        minx_r1 = min([c[0] for c in r1.coords])
        maxx_r1 = max([c[0] for c in r1.coords])
        miny_r1 = min([c[1] for c in r1.coords])
        maxy_r1 = max([c[1] for c in r1.coords])
        minx_r2 = min([c[0] for c in r2.coords])
        maxx_r2 = max([c[0] for c in r2.coords])
        miny_r2 = min([c[1] for c in r2.coords])
        maxy_r2 = max([c[1] for c in r2.coords])
        w1 = maxx_r1 - minx_r1
        h1 = maxy_r1 - miny_r1
        w2 = maxx_r2 - minx_r2
        h2 = maxy_r2 - miny_r2
        l1 = abs( (minx_r1+maxx_r1)/2 - (minx_r2+maxx_r2)/2 )
        l2 = abs( (miny_r1+maxy_r1)/2 - (miny_r2+maxy_r2)/2 )
        if l1<=(w1+w2)/2 and l2<=(h1+h2)/2:
            return True
        return False



            


class QiluAnno():
    def __init__(self, name, shape, coordinates, tumor_grade='', tumor_type='', stroke_color='', stroke_width=''):
        self.name = name
        self.shape = shape
        self.coords= coordinates #[[x1,y1], [x2,y2]]
        self.tumor_grade = tumor_grade
        self.tumor_type = tumor_type
        self.stroke_color = stroke_color
        self.stroke_width = stroke_width
        self.parent = None
        self.children = None

    @classmethod
    def from_dict(cls, d):
        name = d['name']
        shape = d['shape']
        stroke_color = d['strokeColor']
        stroke_width = d['strokeWidth']
        tumor_type = d['tumorType']
        tumor_grade = d['tumorGrade']
        coords = d['Coordinates']
        coords.sort(key=lambda x:int(x['order']))
        coords = [[round(float(c['x'])), round(float(c['y']))] for c in coords]
        return cls(name, shape, coords, tumor_grade, tumor_type, stroke_color, stroke_width)

    @staticmethod
    def set_relation(parent ,child):
        child.parent = parent
        if parent.children == None:
            parent.children = [child]
        else:
            parent.children.append(child)

    def find_max_relation(parents, child):
        max_parent = None
        max_count = 0
        for parent in parents:
            count = 0
            for point in child.coords:
                if QiluAnno.rect_has_point(parent.coords, point):
                    count += 1
            if count>max_count:
                max_count = count
                max_parent = parent
        return max_parent

    def rect_has_point(coords, point):
        # coords: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        # point: [x,y]
        minx = min([c[0] for c in coords])
        miny = min([c[1] for c in coords])
        maxx = max([c[0] for c in coords])
        maxy = max([c[1] for c in coords])
        x,y = point
        if x>minx and x<maxx and y>miny and y<maxy:
            return True
        return False



if __name__ == '__main__':
    #import glob
    #import os
    #xml_dir = '/mnt/md0/_datasets/BCa_QiLu/annotation_20210112'
    #xml_paths = glob.glob(f'{xml_dir}/*.xml')
    #kfb_dir = '/mnt/md0/_datasets/BCa_QiLu/kfb_unnormal'
    #save_dir = '/tmp/Qilu'
    #for xml_path in xml_paths:
    #    name = xml_path.split('/')[-1].replace('.xml', '')
    #    kfb_path = f'{kfb_dir}/{name}.kfb'
    #    try:
    #        wsi = QiluWSI(kfb_path, xml_path)
    #    except Exception as e:
    #        print(e)

        #tar_dir = f'{save_dir}/{name}'
        #os.makedirs(tar_dir, exist_ok=True)
        #scale = 1
        #wsi.save_rects(scale, tar_dir)
        #wsi.save_annos(scale, tar_dir)
        #break
    import time
    start = time.time()
    kfb_path = '/raid10/_datasets/Breast/QiLu/Tumor/04703.15.kfb'
    xml_path = '/raid10/_datasets/Breast/QiLu/Annos/Annotation20210817/04703.15-4306.xml'
    wsi = QiluWSI(kfb_path, xml_path, curr_mag=20)
    wsi.save_rect_mask('/tmp')
    end = time.time()
    print('time:', (end-start))





