import multiprocessing
from multiprocessing import Pool, Manager
from itertools import repeat
import os
import glob

def getPathsBySuffix(src_dir, src_suffix):
    paths = [p for p in glob.glob(f'{src_dir}/*{src_suffix}')  if p.endswith('_mask.png')==False]
    if len(paths)==0:
        paths = [p for p in glob.glob(f'{src_dir}/*/*{src_suffix}')  if p.endswith('_mask.png')==False]
    if len(paths)==0:
        paths = [p for p in glob.glob(f'{src_dir}/*/*/*{src_suffix}')  if p.endswith('_mask.png')==False]
    if len(paths)==0:
        paths = [p for p in glob.glob(f'{src_dir}/*/*/*/*{src_suffix}')  if p.endswith('_mask.png')==False]
    paths = sorted(paths)
    return paths

def getAllFiles(src_dir, ext=None):
    if src_dir.__class__ is list:
        matches = []
        for _dir in src_dir:
            for root, dirnames, filenames in os.walk(_dir):
                for filename in filenames:
                    matches.append(os.path.join(root, filename))
    else:
        matches = []
        for root, dirnames, filenames in os.walk(src_dir):
            for filename in filenames:
                matches.append(os.path.join(root, filename))
    return matches

def getAllFilesBySuffix(src_dir, ext):
    if src_dir.__class__ is list:
        matches = []
        for _dir in src_dir:
            for root, dirnames, filenames in os.walk(_dir):
                for filename in filenames:
                    if filename.endswith((ext)):
                        matches.append(os.path.join(root, filename))
    else:
        matches = []
        for root, dirnames, filenames in os.walk(src_dir):
            for filename in filenames:
                if filename.endswith((ext)):
                    matches.append(os.path.join(root, filename))
    return matches

def multiRunStarmap(f, *args):
    p = Pool(multiprocessing.cpu_count()//2)
    n = 0
    args = list(args)
    for i in range(len(args)):
        if type(args[i]) == list:
            n = len(args[i])
            break

    for i in range(len(args)):
        if type(args[i]) != list:
            args[i] = repeat(args[i], n)

    res = p.starmap(f, zip(*args))
    p.close()
    return res

def multiRunStarmapN(ncpu, f, *args):
    p = Pool(ncpu)
    n = 0
    args = list(args)
    for i in range(len(args)):
        if type(args[i]) == list:
            n = len(args[i])
            break

    for i in range(len(args)):
        if type(args[i]) != list:
            args[i] = repeat(args[i], n)

    res = p.starmap(f, zip(*args))
    p.close()
    return res

def multiRunOneSrcTask(fun, src_dir, src_suffix, tar_dir, tar_ext, params):
    os.makedirs(tar_dir, exist_ok=True)
    src_paths = getPathsBySuffix(src_dir, src_suffix)
    debug = params.get('debug')
    if debug!=None:
        return fun(src_paths[0], tar_dir, tar_ext, params)
    return multiRunStarmap(fun, src_paths, tar_dir, tar_ext, params)

def multiRunTwoSrcDirTask(fun, a_dir, a_suffix, b_dir, b_suffix, tar_dir, tar_suffix, params):
    os.makedirs(tar_dir, exist_ok=True)
    a_paths = getPathsBySuffix(a_dir, a_suffix)
    return multiRunStarmap(fun, a_paths, a_suffix, b_dir, b_suffix, tar_dir, tar_suffix, params)
    

def f(a,b):
    return a+b
if __name__ == '__main__':
    a = [1,2,3,4,5]
    b = [5,6,7,8,9]
    res = multiRunStarmap(f, a,b)
    print(res)
    a = [1,2,3,4,5]
    b = 5
    res = multiRunStarmap(f, a,b)
    print(res)
