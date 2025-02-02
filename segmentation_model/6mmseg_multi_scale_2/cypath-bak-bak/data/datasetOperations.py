import glob
import os
import sys
from PIL import Image

def slink_difference(small_src, large_src, link_tar):
    os.makedirs(link_tar, exist_ok=True)
    A = [os.path.basename(path) for path in glob.glob(f'{small_src}/*')]
    for path in glob.glob(f'{large_src}/*'):
        basename = os.path.basename(path)
        if basename not in A:
            os.symlink(path, path.replace(large_src, link_tar))

def resize(path, src_dir, src_ext, tar_dir, tar_ext, params):
    psize = params.get('psize')
    scale = params.get('scale')

    a = Image.open(path)
    if psize != None:
        b = a.resize((psize,psize))
    elif scale != None:
        w,h = a.size
        b = a.resize((w//scale,h//scale)) 
    else:
        return
    save_path = path.replace(src_dir, tar_dir)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    b.save(save_path)


if __name__ == '__main__':
    small_src = sys.argv[1]
    large_src = sys.argv[2]
    link_tar = sys.argv[3]
    slink_difference(small_src, large_src, link_tar)
