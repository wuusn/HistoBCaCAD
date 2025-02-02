import os
import cv2
import glob
import argparse
from morphologyProcessing import maskFile2JsonFineFile

def parse_args():
    parser = argparse.ArgumentParser(description='merge multi res results')
    parser.add_argument('--save-dir', help='save dir')
    parser.add_argument('--src-dir', help='save dir')
    args = parser.parse_args()
    return args

def merge(src_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    ori_paths = glob.glob(f'{src_dir}/*.jpg')
    for ori_path in ori_paths:
        basename = os.path.basename(ori_path)
        name = basename.replace('.jpg', '')
        src_paths = glob.glob(f'{src_dir}/{name}-*.png')

        final_mask = cv2.imread(src_paths[0], 0)
        for src_path in src_paths:
            mask = cv2.imread(src_path, 0)
            try:
                final_mask[mask!=255]=0
            except:
                print(src_path)
        cv2.imwrite(f'{save_dir}/{name}.png', final_mask)
        maskFile2JsonFineFile(f'{save_dir}/{name}.png', '.png', 16, save_dir, '.json')



if __name__ == '__main__':
    args = parse_args()
    import time
    start = time.time()
    merge(args.src_dir, args.save_dir)
    end = time.time()
    print('done:', (end-start))
