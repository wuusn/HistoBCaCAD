import os
import sys
import yaml
import glob
from cypath.data.multiRun import getAllFiles, getAllFilesBySuffix

def linkAllFilesToTarL1(paths, tar_dir, params):
    os.makedirs(tar_dir, exist_ok=True)
    for path in paths:
        basename = os.path.basename(path)
        os.symlink(os.path.realpath(path), f'{tar_dir}/{basename}')
        mask_ext = params.get('mask_ext')
        src_ext = params.get('src_ext')
        if mask_ext:
            path = path.replace(src_ext, mask_ext)
            basename = basename.replace(src_ext, mask_ext)
            os.symlink(os.path.realpath(path), f'{tar_dir}/{basename}')

def linkAllFilesFromSrcToTarL0(src_dir, tar_dir, params):
    os.makedirs(tar_dir, exist_ok=True)
    ext = params.get('ext')
    if ext:
        files = getAllFilesBySuffix(src_dir, ext)
    else:
        files = getAllFiles(src_dir)

    for path in files:
        basename = os.path.basename(path)
        os.symlink(os.path.realpath(path), f'{tar_dir}/{basename}')

def linkAllFilesFromSrcsToTarL1(src_dirs, tar_dir, params):
    os.makedirs(tar_dir, exist_ok=True)
    for src_dir in src_dirs:
        files = getAllFiles(src_dir)
        for path in files:
            basename = os.path.basename(path)
            os.symlink(os.path.realpath(path), f'{tar_dir}/{basename}')

def linkAllFilesFromSrcToTarL2(src_dir, tar_dir, params):
    split = params.get('split', '-')
    idxs = params.get('idxs')
    idxs = str(idxs)
    idxs = idxs.split(',')
    os.makedirs(tar_dir, exist_ok=True)
    files = getAllFiles(src_dir)
    for path in files:
        basename = os.path.basename(path)
        ext = basename.split('.')[-1]
        name = basename.replace(f'.{ext}', '')
        splits = name.split(split)
        group = '-'.join([splits[int(i)] for i in idxs])
        save_dir = f'{tar_dir}/{group}'
        os.makedirs(save_dir, exist_ok=True)
        os.symlink(os.path.realpath(path), f'{save_dir}/{basename}')

def linkAllFilesFromSrcsToTarL3(src_dirs, tar_dir, split, idxs1, idxs2):
    split = params.get('split', '-')
    idxs1 = params.get('idxs1')
    idxs1 = idxs1.split(',')
    idxs2 = params.get('idxs2')
    idxs2 = idxs2.split(',')
    ext = params.get('src_ext')
    os.makedirs(tar_dir, exist_ok=True)
    for src_dir in src_dirs:
        files = getAllFiles(src_dir)
        for path in files:
            basename = os.path.basename(path)
            name = basename.replace(ext, '')
            splits = name.split(split)
            group1 = '-'.join([splits[int(i)] for i in idxs1])
            group2 = '-'.join([splits[int(i)] for i in idxs2])
            save_dir = f'{tar_dir}/{group1}/{group2}'
            os.makedirs(save_dir, exist_ok=True)
            os.symlink(os.path.realpath(path), f'{save_dir}/{basename}')

def linkTemplate(src_dir, src_suffix, tem_dir, tem_suffix, tar_dir):
    os.makedirs(tar_dir, exist_ok=True)
    tem_paths = glob.glob(f'{tem_dir}/*{tem_suffix}')
    for tem_path in tem_paths:
        name = tem_path.split('/')[-1].replace(tem_suffix, '')
        src_path = f'{src_dir}/{name}{src_suffix}'
        os.symlink(os.path.realpath(src_path), f'{tar_dir}/{name}{src_suffix}')

def linkL122(src_dir, tar_dir, params):
    split = params.get('split', '-')
    idxs = params.get('idxs')
    idxs = idxs.split(',')
    ext = params.get('src_ext')
    set_dirs = glob.glob(f'{src_dir}/*')
    for set_dir in set_dirs:
        tar_dir2 = set_dir.replace(src_dir, tar_dir)
        os.makedirs(tar_dir2, exist_ok=True)
        for path in glob.glob(f'{set_dir}/*'):
            basename = os.path.basename(path)
            name = basename.replace(ext, '')
            splits = name.split(split)
            group = '-'.join([splits[int(i)] for i in idxs])
            save_dir = f'{tar_dir2}/{group}'
            os.makedirs(save_dir, exist_ok=True)
            os.symlink(os.path.realpath(path), f'{save_dir}/{basename}')

if __name__ == '__main__':
    yaml_path = sys.argv[1]
    with open(yaml_path, 'r') as f:
        param_sets = yaml.safe_load(f)
    for set_name, param in param_sets.items():
        fun_name = param.get('funName')
        src_dir = param.get('src_dir')
        tar_dir = param.get('tar_dir')
        fun = globals()[fun_name]
        fun(src_dir, tar_dir, param)
    #src_dir = '/raid10/_datasets/Oral/TMA/IHCHE/he_result_xujun_mask'
    #src_suffix = '_pred.png'
    #tem_dir = '/raid10/_datasets/Oral/TMA/IHCHE/tissue_epi_0.3'
    #tem_suffix = '_real_A_mask.png'
    #tar_dir = '/raid10/_datasets/Oral/TMA/IHCHE/he_result_xujun_0.3'
    #linkTemplate(src_dir, src_suffix, tem_dir, tem_suffix, tar_dir)
    #src_dir = '/raid10/_datasets/Oral/TMA/IHCHE/correct_pred_mask'
    #tar_dir = '/raid10/_datasets/Oral/TMA/IHCHE/correct_pred_mask_0.3'
    #linkTemplate(src_dir, src_suffix, tem_dir, tem_suffix, tar_dir)
