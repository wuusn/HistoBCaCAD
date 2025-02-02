import os
import glob
from cypath.data.multiRun import multiRunStarmap, getPathsBySuffix
import yaml
import sys

def checkrm(path, kB=300):
    size = os.stat(path).st_size
    if size < kB*1024:
        os.remove(path)

if __name__ == '__main__':
     yaml_path = sys.argv[1]
     with open(yaml_path, 'r') as f:
         param_sets = yaml.safe_load(f)
     for set_name, param in param_sets.items():
         src_dir = param.get('src_dir')
         src_ext = param.get('src_ext')
         kB = param.get('size')
         paths = getPathsBySuffix(src_dir, src_ext)
         multiRunStarmap(checkrm, paths, kB)
