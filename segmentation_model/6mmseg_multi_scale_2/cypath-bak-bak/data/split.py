import os
import glob
import sys
from sklearn.model_selection import train_test_split
import cypath.data.link
from cypath.data.link import *
from cypath.data.multiRun import getAllFiles, getAllFilesBySuffix
import yaml

def randomSplit(lists, ratio):
    model_lists, test_lists= train_test_split(lists, test_size=ratio, random_state=56)
    return model_lists, test_lists

def randomSplitLS(src_dir, tar_dir, ratio, phrases, label):
    case_dirs = glob.glob(f'{src_dir}/*')
    train_dirs, test_dirs = train_test_split(case_dirs, test_size=ratio, random_state=56)
    dirs = [train_dirs, test_dirs]
    for i in range(2):
        tmp_dirs = dirs[i]
        phrase = phrases[i]
        phrase_dir = f'{tar_dir}/{phrase}/{label}'
        os.makedirs(phrase_dir, exist_ok=True)
        for tmp_dir in tmp_dirs:
            link_dir = tmp_dir.replace(src_dir, phrase_dir)
            os.symlink(os.path.realpath(tmp_dir), link_dir)

def randomRegionSplitLS(src_dir, tar_dir, ratio, phrases):
    case_dirs = glob.glob(f'{src_dir}/normal/*')
    tmp_dirs = glob.glob(f'{src_dir}/tumor/*/*')
    case_dirs.extend(tmp_dirs)
    train_dirs, test_dirs = train_test_split(case_dirs, test_size=ratio, random_state=56)
    dirs = [train_dirs, test_dirs]
    for i in range(2):
        tmp_dirs = dirs[i]
        phrase = phrases[i]
        phrase_dir = f'{tar_dir}/{phrase}'
        os.makedirs(phrase_dir, exist_ok=True)
        for tmp_dir in tmp_dirs:
            name = tmp_dir.split('/')[-2]+'_'+tmp_dir.split('/')[-1]
            link_dir = f'{phrase_dir}/{name}'
            os.symlink(os.path.realpath(tmp_dir), link_dir)

def filterFileSize(_dir, params):
    min_size = params.get(f'min_size', 0)
    max_size = params.get(f'max_size', float('inf'))
    filesize = len(glob.glob(f'{_dir}/*'))
    if filesize<min_size or filesize>=max_size:
        return True
    else:
        return False

def getOutlierWSISet(funs, dirs, params):
    wsis = set()
    for _dir in dirs:
        for fun in funs:
            if fun(_dir, params):
                wsis.add(_dir.split('/')[-1].split('-')[0])
                break
    return wsis


def checkOutlierTemplate(name, template_dir):
    if os.path.exists(f'{template_dir}/{name}'):
        return True
    else:
        return False

def manualSplit(src_dir, tar_dir, ratio, phrases, label):
    case_dirs = glob.glob(f'{src_dir}/*')
    outlier_dirs = []
    model_dirs = []
    for case_dir in case_dirs:
        type_grade_dirs = glob.glob(f'{case_dir}/*')
        outlier = False
        for type_grade_dir in type_grade_dirs:
            files = glob.glob(f'{type_grade_dir}/*')
            filesize = len(files)
            if checkOutlier(filesize):
                outlier = True
                break
        if outlier:
            outlier_dirs.append(case_dir)
        else:
            model_dirs.append(case_dir)
      
    os.makedirs(f'{tar_dir}/outlier',exist_ok=True)
    for outlier_dir in outlier_dirs:
        link_dir = outlier_dir.replace(src_dir, f'{tar_dir}/outlier')
        os.symlink(os.path.realpath(outlier_dir), link_dir)

    train_dirs, test_dirs = train_test_split(model_dirs, test_size=ratio, random_state=56)
    dirs = [train_dirs, test_dirs]
    for i in range(2):
        tmp_dirs = dirs[i]
        phrase = phrases[i]
        phrase_dir = f'{tar_dir}/{phrase}/{label}'
        os.makedirs(phrase_dir, exist_ok=True)
        for tmp_dir in tmp_dirs:
            link_dir = tmp_dir.replace(src_dir, phrase_dir)
            os.symlink(os.path.realpath(tmp_dir), link_dir)

def randomPatchSplit(src_dir, tar_dir, ratio, phrases):
    tmp_files = glob.glob(f'{src_dir}/*/*/*.png')
    tmp_files2 = glob.glob(f'{src_dir}/*/*/*/*.png')
    tmp_files.extend(tmp_files2)
    train_paths, val_paths = train_test_split(tmp_files, test_size=ratio, random_state=56)
    paths = [train_paths, val_paths]
    for i in range(2):
        tmp_paths = paths[i]
        phrase = phrases[i]
        j = 0
        for tmp_path in tmp_paths:
            splits = tmp_path.split('/')
            j+=1
            name = splits[-1]
            name = name.replace('.png', f'_{j}.png')
            if len(splits)>9:
                label = splits[-2]
                label = label.split('-')[0]
            else:
                label = splits[-3]
            phrase_dir = f'{tar_dir}/{phrase}/{label}'
            os.makedirs(phrase_dir, exist_ok=True)
            tar_path = f'{phrase_dir}/{name}'
            os.symlink(os.path.realpath(tmp_path), tar_path)

def manualSplitWithOutlierTemplate(src_dir, tar_dir, outlier_template, ratio, phrases, label):
    case_dirs = glob.glob(f'{src_dir}/*')
    outlier_dirs = []
    model_dirs = []
    for case_dir in case_dirs:
        name = case_dir.split('/')[-1]
        if checkOutlierTemplate(name, outlier_template):
            outlier_dirs.append(case_dir)
        else:
            model_dirs.append(case_dir)
      
    os.makedirs(f'{tar_dir}/outlier',exist_ok=True)
    for outlier_dir in outlier_dirs:
        link_dir = outlier_dir.replace(src_dir, f'{tar_dir}/outlier')
        os.symlink(os.path.realpath(outlier_dir), link_dir)

    train_dirs, test_dirs = train_test_split(model_dirs, test_size=ratio, random_state=56)
    dirs = [train_dirs, test_dirs]
    for i in range(2):
        tmp_dirs = dirs[i]
        phrase = phrases[i]
        phrase_dir = f'{tar_dir}/{phrase}/{label}'
        os.makedirs(phrase_dir, exist_ok=True)
        for tmp_dir in tmp_dirs:
            link_dir = tmp_dir.replace(src_dir, phrase_dir)
            os.symlink(os.path.realpath(tmp_dir), link_dir)

def randomPatchSplitGradeLabel(src_dir, tar_dir, ratio, phrases):
    tmp_files = glob.glob(f'{src_dir}/*/*/*.png')
    tmp_files2 = glob.glob(f'{src_dir}/*/*/*/*.png')
    tmp_files.extend(tmp_files2)
    train_paths, val_paths = train_test_split(tmp_files, test_size=ratio, random_state=56)
    paths = [train_paths, val_paths]
    for i in range(2):
        tmp_paths = paths[i]
        phrase = phrases[i]
        j = 0
        for tmp_path in tmp_paths:
            splits = tmp_path.split('/')
            j+=1
            name = splits[-1]
            name = name.replace('.png', f'_{j}.png')
            if len(splits)>9:
                label = splits[-2]
                _type = label.split('-')[0]
                grade = label.split('-')[1]
            else:
                _type = splits[-3]
                grade = _type
            phrase_dir = f'{tar_dir}/{phrase}/{_type}/{grade}'
            os.makedirs(phrase_dir, exist_ok=True)
            tar_path = f'{phrase_dir}/{name}'
            os.symlink(os.path.realpath(tmp_path), tar_path)

def getDictListFromSet(_dict, _set):
    _list=[]
    for k in list(_set):
        _list.extend(_dict[k])
    return _list

def split_workflow(src_dir, tar_dir, params):

    # filter outlier
    regions = glob.glob(f'{src_dir}/*')
    wsi_set=set()
    wsi_dict=dict()
    for region in regions:
        region_name = region.split('/')[-1]
        wsi_name = region_name.split('-')[0]
        wsi_set.add(wsi_name)
        if wsi_dict.get(wsi_name) is None:
            wsi_dict[wsi_name] = [region]
        else:
            wsi_dict[wsi_name].append(region)

    outlierfuns = params.get('outlierFuns', 'filterFileSize')
    funs = [globals()[funname] for funname in outlierfuns.split(',')]
    outlier_set = getOutlierWSISet(funs, regions, params)
    wsi_set = wsi_set - outlier_set

    regions = getDictListFromSet(wsi_dict, wsi_set)

    # link outliers
    outliers = getDictListFromSet(wsi_dict, outlier_set)
    linkAllFilesFromSrcsToTarL1(outliers, tar_dir+os.sep+'outlier', params)

    # link models and tests
    model_wsi, test_wsi = randomSplit(list(wsi_set), 0.2)
    models = getDictListFromSet(wsi_dict, set(model_wsi))
    tests = getDictListFromSet(wsi_dict, set(test_wsi))
    linkAllFilesFromSrcsToTarL1(models, tar_dir+os.sep+'model', params)
    linkAllFilesFromSrcsToTarL1(tests, tar_dir+os.sep+'test', params)

    # link train and val
    src_ext = params.get('src_ext')
    if src_ext:
        model_paths = getAllFilesBySuffix(models, src_ext)
    else:
        model_paths = getAllFiles(models)

    trains, vals = randomSplit(model_paths, 0.2)

    linkAllFilesToTarL1(trains, tar_dir+os.sep+'train', params)
    linkAllFilesToTarL1(vals, tar_dir+os.sep+'val', params)


if __name__ == '__main__':
    yaml_path = sys.argv[1]
    with open(yaml_path, 'r') as f:
        param_sets = yaml.safe_load(f)
    for set_name, param in param_sets.items():
        src_dir = param.get('src_dir')
        tar_dir = param.get('tar_dir')
        split_workflow(src_dir, tar_dir, param)
# 10x small dataset for tis/it classification
    #src_dir = '/ssd/qilu20210715_10x/10x512NormalPatchOnInOut'
    #tar_dir = '/ssd/qilu20210715_10x/normal_tis_it_2/wsi_wise'
    #ratio = 0.2
    #phrases = ['model', 'test']
    #label = 'normal'
    #randomSplitLS(src_dir, tar_dir, ratio, phrases, label)

    #src_dir = '/ssd/qilu20210715_10x/10x512Anno20210715PatchOverLap50'
    #tar_dir = '/ssd/qilu20210715_10x/normal_tis_it_2/wsi_wise'
    #ratio = 0.1
    #phrases = ['model', 'test']
    #label = 'tumor'
    #manualSplit(src_dir, tar_dir, ratio, phrases, label)

    #src_dir = '/ssd/qilu20210715_10x/normal_tis_it_2/wsi_wise/model'
    #tar_dir = '/ssd/qilu20210715_10x/normal_tis_it_2/patch_wise'
    #ratio = 0.2
    #phrases = ['train', 'val']
    #randomPatchSplit(src_dir, tar_dir, ratio, phrases)

    #src_dir = '/ssd/qilu20210715_10x/normal_tis_it_2/wsi_region_wise/model'
    #tar_dir = '/ssd/qilu20210715_10x/normal_tis_it_2/region_wise'
    #ratio = 0.2
    #phrases = ['train', 'val']
    #randomRegionSplitLS(src_dir, tar_dir, ratio, phrases)
