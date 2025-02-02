import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import sys
import glob
sys.path.insert(0, '/home/yuxin/bme/BCaCAD/model/6mmseg_multi_scale_2')
from cypath.data.wsi import WSI
from cypath.fast import *
from im_state.aa import getTissueMask
from PIL import Image
import mmseg
#from mmcv.utils import Config
#from mmcv.config import Config
from mmengine.config import Config
import argparse
import gc
import cv2
import pandas as pd
import resource
from morphologyProcessing import post_processing, gaussian_ostu, saveMorphoProcess
import gc

def convertConfigFile(config_path, psize, root, save_tmp_dir):
    cfg = Config.fromfile(config_path)
    cfg.data.test.data_root = root
    cfg.data.test.img_dir = ''
    cfg.data.test.ann_dir = ''
    cfg.data.test.pipeline[1].img_scale = (psize, psize)
    basename = os.path.basename(config_path)
    save_path = f'{save_tmp_dir}/{basename}'
    cfg.dump(save_path)
    return save_path

def checkResults(path):
    if os.path.exists(path):
        return True
    return False

def testOneWSI(pth_path, config_path, wsi_path, save_dir, tmp_dir, max_mag, curr_mag, psize, save_scale, tail_name, ncpu):
    ext = '.'+wsi_path.split('.')[-1]
    name = os.path.basename(wsi_path).replace(ext, '')

    if checkResults(f'{save_dir}/{name}{tail_name}.png'):
        return None, None

    tmp_dir = tmp_dir+f'/{name}'
    os.makedirs(tmp_dir, exist_ok=True)
    wsi2Patch(wsi_path, tmp_dir, '.jpg', max_mag, curr_mag, psize, ncpu)

    tmp_config_path = convertConfigFile(config_path, psize, tmp_dir, tmp_dir)
    #test_cmd = f'./tools/dist_test.sh {tmp_config_path} {pth_path} 2 --format-only --eval-options imgfile_prefix={tmp_dir}'
    test_cmd = f'python tools/test.py {tmp_config_path} {pth_path} --format-only --eval-options imgfile_prefix={tmp_dir}'
    os.system(test_cmd)

    wsi = WSI(wsi_path, max_mag, curr_mag)
    W,H = wsi.getSize()
    #print(W,H)
    M = mergePatch(name, W,H,1, psize, tmp_dir,  save_dir, '.png', None, save_scale, '-', -2, -1)
    O = mergePatch(name, W,H,3, psize, tmp_dir,  save_dir, '.jpg', None, save_scale, '-', -2, -1)
    M = M.astype(np.uint8)
    im = Image.fromarray(M)
    mask_path = f'{save_dir}/{name}{tail_name}.png'
    im.save(mask_path)
    im = Image.fromarray(O)
    ori_path = f'{save_dir}/{name}.jpg'
    im.save(ori_path)
    return mask_path, ori_path

def parse_args():
    parser = argparse.ArgumentParser(description='Inference a segmentor')
    parser.add_argument('config', help='config file path')
    parser.add_argument('ckpt', help='checkpoint config file path')
    parser.add_argument('--wsi-dir', help='wsi dir')
    parser.add_argument('--wsi-ext', help='wsi file type, with .')
    parser.add_argument('--save-dir', help='save dir')
    parser.add_argument('--tmp-dir', help='tmp dir', default='/tmp/usjgiaugioa')
    parser.add_argument('--max-mag', help='max magnification of wsi', type=int, default=40)
    parser.add_argument('--curr-mag', help='the magnification of the model', type=int, default=10)
    parser.add_argument('--save-mag', help='the downscale from curr_mag for saving results', type=float, default=2)
    parser.add_argument('--tail-name', help='tailname for test', type=str, default='')
    parser.add_argument('--psize', help='the patch size fro cropping wsi', type=int, default=512)
    parser.add_argument('--ncpu', help='n workers to do wsi2patch', type=int, default=10)
    args = parser.parse_args()
    return args

def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 / 4 * 3, hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

if __name__ == '__main__':
    import time
    #memory_limit()
    start = time.time()

    args = parse_args()
    print(args)
    os.system(f'rm -rf {args.tmp_dir}')
    if os.path.isfile(args.wsi_dir):
        if args.wsi_dir.endswith('.xlsx'):
            df = pd.read_excel(args.wsi_dir)
            wsi_paths = list(df.filepath)
        else:
            wsi_paths = [args.wsi_dir]
    elif args.wsi_ext:
        wsi_paths = sorted(glob.glob(f'{args.wsi_dir}/*{args.wsi_ext}'))
    else:
        wsi_paths = sorted(glob.glob(f'{args.wsi_dir}/*'))

    save_scale = args.curr_mag // args.save_mag
    save_scale = int(save_scale)

    mask_paths = []
    ori_paths = []
    #wsi_paths = wsi_paths[:1]
    for wsi_path in wsi_paths:
        gc.collect()
        print()
        print(os.path.basename(wsi_path))
        mask_path, ori_path = testOneWSI(args.ckpt, args.config, wsi_path, args.save_dir, args.tmp_dir, args.max_mag, args.curr_mag, args.psize, save_scale, args.tail_name, args.ncpu)
        if mask_path != None:
            mask_paths.append(mask_path)
            ori_paths.append(ori_path)
        gc.collect()
        os.system(f'rm -rf {args.tmp_dir}')
    for i in range(len(mask_paths)):
        mask_path = mask_paths[i]
        ori_path = ori_paths[i]
        saveMorphoProcess(mask_path, ori_path, '.png', args.save_dir, '.png')

    end = time.time()
    print()
    print('done! time:', (end-start)/60, 'min')
    print('done! time:', (end-start), 's')
