import cv2
import numpy as np
from skimage.morphology import remove_small_objects, binary_opening, disk
from skimage import io, color, img_as_ubyte
import skimage
import geojson
from PIL import Image
import PIL
Image.MAX_IMAGE_PIXELS = None
#from histomicstk.saliency import tissue_detection# import get_tissue_mask
import random
import glob
import os
from multiprocessing import Process
import subprocess

def getMd5(path):
    output = subprocess.check_output(['md5sum', path])
    output = output.decode('utf-8')
    output = output.split(' ')[0]
    #print(output)
    return output

def post_processing(mask, scale=1):
    area_base = 200
    #area_base = 100
    min_base = 300
    kernel_size = 5

    #print(np.unique(mask))
    mask1= mask / 255 if np.max(mask)>1 else mask
    mask2= mask == 255 if np.max(mask)>1 else mask
    mask3 = mask >= 127
    #print(np.sum(mask1.astype(bool)))
    #print(np.sum(mask2.astype(bool)))

    #mask = mask2 # only keep 255
    mask = mask1 # keep all
    #mask = mask3 # keep part
    mask = mask.astype(bool)
    area_thresh = area_base//scale
    mask_opened = remove_small_objects(mask, min_size=area_thresh)
    mask_removed_area = ~mask_opened & mask
    mask = mask_opened > 0

    min_size = min_base//scale
    img_reduced = skimage.morphology.remove_small_holes(mask, area_threshold=min_size)
    img_small = img_reduced & np.invert(mask)
    mask = img_reduced
    mask = mask.astype(np.uint8)
    kernel = np.ones((kernel_size,kernel_size), dtype=np.uint8)
    new = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return new

def get_tissue_mask(img):
    labeled, mask = tissue_detection.get_tissue_mask(
                    img, deconvolve_first=False,
                    n_thresholding_steps=1, sigma=0, min_size=30)
    return labeled>0
    return mask # bool type


def gaussian_ostu(img):
    blur = cv2.GaussianBlur(img, (5,5), 0)
    ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = ~th
    #th = np.ones(blur.shape)*255
    #cv2.imwrite(f'/mnt/raid5/_datasets/Breast/Breast_model_results/shandaer/02028-2018-4_otsu.png', th)
    return th # 255

def morphoProcess10x(mask):
    return post_processing(mask, 1/4)

def saveMorphoProcess10x(src_path, src_suffix, tar_dir, tar_suffix):
    name = src_path.split('/')[-1].replace(src_suffix, '')
    mask = cv2.imread(src_path, 0)
    res = morphoProcess10x(mask)
    res = res * 255 if np.max(res) < 2 else res
    cv2.imwrite(f'{tar_dir}/{name}{tar_suffix}', res)

def saveMorphoProcess(src_path, src_path2, src_suffix, tar_dir, tar_suffix):
    name = src_path.split('/')[-1].replace(src_suffix, '')
    mask = cv2.imread(src_path, 0)
    gray_img = cv2.imread(src_path2, 0)
    otsu = gaussian_ostu(gray_img)
    mask[~(otsu==255)] = 0

    res = post_processing(mask)

    res = res * 255 if np.max(res) < 2 else res
    cv2.imwrite(f'{tar_dir}/{name}{tar_suffix}', res)

def mask2Json(mask, upscale, adjust):
    #mask= mask / 255 if np.max(mask)>1 else mask
    #bimg, contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cnts=[]
    for cnt in contours:
        if len(cnt)<4: continue
        arclen = cv2.arcLength(cnt, True)
        epsilon = arclen * adjust
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

def mask2JsonFine(mask, upscale):
    adjust = 0.0002
    return mask2Json(mask, upscale, adjust)

def maskFile2JsonFineFile(maskpath, mask_suffix, upscale, tar_dir, tar_suffix):
    name = maskpath.split('/')[-1].replace(mask_suffix, '')
    mask = cv2.imread(maskpath, 0)
    json = mask2JsonFine(mask, upscale)
    with open(f'{tar_dir}/{name}{tar_suffix}', 'w') as f:
        geojson.dump(json, f)

def cvtFileMask2Json(src_path, src_suffix, tar_dir, tar_suffix):
    return maskFile2JsonFineFile(src_path, src_suffix, 1, tar_dir, tar_suffix)

def drawJsonPathOnImagePath(json_path, json_suffix, src_path, src_suffix, tar_dir, tar_suffix):
    name = json_path.split('/')[-1].replace(json_suffix, '')
    with open(json_path) as f:
        gj = geojson.load(f)
    features = gj[0]
    cnts = features['geometry']['coordinates']
    cnts = np.array([cnts], dtype=np.uint8)
    #cnts = cnts.unsqueeze()
    io = cv2.imread(src_path)
    cv2.drawContours(io, cnts, -1, (0, 255, 0), 3)
    cv2.imwrite(f'{tar_dir}/{name}{tar_suffix}', io)

def drawMaskContourPathOnImagePath(mask_path, mask_suffix, src_path, src_suffix, tar_dir, tar_suffix):
    name = mask_path.split('/')[-1].replace(mask_suffix, '')
    mask = cv2.imread(mask_path, 0)
    bimg, contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    io = cv2.imread(src_path)
    cv2.drawContours(io, contours, -1, (0, 255, 0), 3)
    #cv2.imwrite(f'{tar_dir}/{name}{tar_suffix}', io)
    Image.fromarray(io).save(f'{tar_dir}/{name}{tar_suffix}')

def combineTwoImagePath(a_path, a_suffix, b_dir, b_suffix, save_dir, save_suffix, params):
    scale = params['scale']
    scale = int(scale)
    name = a_path.split('/')[-1].replace(a_suffix, '')

    imA = Image.open(a_path).convert(mode='RGB')
    b_path = f'{b_dir}/{name}{b_suffix}'
    imB = Image.open(b_path).convert(mode='RGB')

    width, height = imA.size
    width = width // 4
    height = height // 4

    total_width = width * 2
    max_height = height

    imA = imA.resize((width, height), PIL.Image.BICUBIC)
    imB = imB.resize((width, height), PIL.Image.BICUBIC)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in [imA, imB]:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]


    #npim = np.array(new_im)
    #npim = npim[...,::-1]
    #cv2.imwrite(f'{save_dir}/{name}{save_suffix}', npim)
    new_im.save(f'{save_dir}/{name}{save_suffix}')

def combineManyImagePath(a_path, a_suffix, save_dir, save_suffix, *dirSuffixPair):
    name = a_path.split('/')[-1].replace(a_suffix, '')
    img_paths = [a_path]
    dirSuffixPair = list(dirSuffixPair)
    i = 0
    while i < len(dirSuffixPair):
        b_dir = dirSuffixPair[i]
        b_suffix = dirSuffixPair[i+1]
        i+=2
        b_path = f'{b_dir}/{name}{b_suffix}'
        img_paths.append(b_path)
    images = [Image.open(path).convert(mode='RGB') for path in img_paths]
    width, height = images[0].size
    total_width = width * len(images)
    x_offset = 0
    new_im = Image.new('RGB', (total_width, height))
    for im in images:
        im = im.resize((width, height), PIL.Image.BICUBIC)
        new_im.paste(im, (x_offset, 0))
        x_offset += width
    new_im.save(f'{save_dir}/{name}{save_suffix}')

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

def rowShowDiffSrcImages(img_amount, row_size, save_dir, *dirSuffixPair):
    dirSuffixPair = list(dirSuffixPair)
    src_dirs = []
    suffixes = []
    i = 0
    while i < len(dirSuffixPair):
        b_dir = dirSuffixPair[i]
        b_suffix = dirSuffixPair[i+1]
        src_dirs.append(b_dir)
        suffixes.append(b_suffix)
        i+=2

    img_paths = []
    for i in range(len(src_dirs)):
        imgs = glob.glob(f'{src_dirs[i]}/*{suffixes[i]}')
        img_paths.append(imgs)

    sample_path = imgs[0]
    sample = Image.open(sample_path)
    w,h = sample.size

    os.makedirs(save_dir, exist_ok=True)
    for j in range(img_amount):
        O = np.ones((h*len(src_dirs),w*row_size,3)).astype(np.uint8)
        for i in range(len(src_dirs)):
            random.shuffle(img_paths[i])
            for k in range(row_size):
                im = Image.open(img_paths[i][k])
                np_im = np.array(im).astype(np.uint8)
                O[i*h:i*h+h,k*w:k*w+w,:] = np_im
                Image.fromarray(O).save(f'{save_dir}/{j}.jpg')




def find_epis(np_im, scale):
    deconvolve_first = False
    n_thresholding_steps = 1
    sigma = 0.
    min_size = 30
    labeled, mask = get_tissue_mask(
                        np_im, deconvolve_first=deconvolve_first,
                                    n_thresholding_steps=n_thresholding_steps, sigma=sigma, min_size=min_size)
    mask = mask.astype(np.uint8)
    cnts = mask2cnts(mask, scale)
    return cnts


if __name__ == '__main__':
    #mask_path = '/mnt/md0/_datasets/OralCavity/share_epi_seg_model/resultHEDifferredMasksMorphoed/OCSCC TMA 1 CK_regist_fp_morphoed.png'
    #mask_suffix = '_fp_morphoed.png'
    #json_path = '/mnt/md0/_datasets/OralCavity/share_epi_seg_model/resultHEDifferredMasksMorphoed/OCSCC TMA 1 CK_regist_fp.json'
    #json_suffix = '_fp.json'
    #src_path = '/mnt/D/Oral/IHC@10x_2/result_refine_regist_removedHE/Oral cavity TMA block 1.png'
    #src_suffix = '.png'
    #tar_dir = '/mnt/D/Oral/aaa'
    #tar_suffix = '_fn_contour.png'
    ##drawJsonPathOnImagePath(json_path, json_suffix, src_path, src_suffix, tar_dir, tar_suffix)
    #drawMaskContourPathOnImagePath(mask_path, mask_suffix, src_path, src_suffix, tar_dir, tar_suffix)
    src_suffix='.png'
    tar_suffix='.png'
    cohorts = ['qingdao', 'shandaer']
    #cohorts = ['shandaer']
    for cohort in cohorts:
        src_dir = f'/mnt/raid5/_datasets/Breast/Breast_model_results/{cohort}/tumor_mask_5'
        save_mask_dir = src_dir+'_127_morph_otsu_tissue'
        tar_dir = save_mask_dir
        #save_json_dir = src_dir+'_json'
        os.makedirs(save_mask_dir, exist_ok=True)
        #os.makedirs(save_json_dir, exist_ok=True)
        src_paths = glob.glob(f'{src_dir}/*{src_suffix}')
        for src_path in src_paths:
            p = Process(target=saveMorphoProcess, args=(src_path, src_suffix, tar_dir, tar_suffix))
            p.start()
            p.join()
    for cohort in cohorts:
        src_dir = f'/mnt/raid5/_datasets/Breast/Breast_model_results/{cohort}/tumor_mask_5'
        tar_dir = src_dir+'_127_json_otsu_tissue'
        src_dir = src_dir+'_127_morph_otsu_tissue'
        os.makedirs(tar_dir, exist_ok=True)
        src_paths = glob.glob(f'{src_dir}/*{src_suffix}')
        for src_path in src_paths:
            p = Process(target=maskFile2JsonFineFile, args=(src_path, src_suffix, 20, tar_dir, '.json'))
            p.start()
            p.join()
