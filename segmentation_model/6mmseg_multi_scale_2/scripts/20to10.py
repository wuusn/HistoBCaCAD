from cypath.data.datasetOperations import resize
from cypath.data.multiRun import multiRunStarmap, getAllFilesBySuffix

if __name__ == '__main__':
    #src_dir = '/raid10/_datasets/Breast/QiLu/20xRegion9918'
    #tar_dir = '/raid10/_datasets/Breast/QiLu/10xRegion9918'
    scale = 2
    params=dict(scale=scale)
    src_ext = '.jpg'
    tar_ext = src_ext
    paths = getAllFilesBySuffix(src_dir, src_ext)
    multiRunStarmap(resize, paths, src_dir, src_ext, tar_dir, tar_ext, params)
    src_ext = '.jpg'
    tar_ext = src_ext
    paths = getAllFilesBySuffix(src_dir, src_ext)
    multiRunStarmap(resize, paths, src_dir, src_ext, tar_dir, tar_ext, params)
