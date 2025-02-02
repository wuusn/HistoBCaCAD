from .getBoxFromAnno import *
from .compareMasks import *
from .multiRun import *
import cv2
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import math

from sklearn.metrics import auc, roc_curve

def evalOneResultAnnoStepThreshold(result, anno, step=.01):
    threshold = np.arange(0,1, step)
    threshold = np.append(threshold, 1.1)
    ms = []
    for t in threshold:
        m = compareMasks(result, anno, t)
        ms.append(m)
    return ms, threshold

def evalOneResultAnno(result, anno):
    return compareMasks(result, anno)

def drawROC(fpr, tpr, roc_auc, save_path= 'roc.png'):
    lw = 2
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(save_path)



def evalOneBoxResultAnno(result, anno):
    box = getBoxFromAnno(anno)
    anno = getRegionOfBox(box, anno)
    result = getRegionOfBox(box, result)
    return evalOneResultAnno(result, anno)

def evalOneResultAnnoWithExclude(result, anno, exclude):
    result = excludeMask(result, exclude)
    anno = excludeMask(anno, exclude)
    return evalOneResultAnno(result, anno)

def evalOneBoxResultAnnoWithExclude(result, anno, exclude):
    result = excludeMask(result, exclude)
    anno = excludeMask(anno, exclude)
    return evalOneBoxResultAnno(result, anno)

def evalOneResultAnnoCheckExcludePath(result, anno, exclude_path):
    if os.path.exists(exclude_path):
        exclude = cv2.imread(exclude_path, 0)
        return evalOneResultAnnoWithExclude(result, anno, exclude)
    else:
        return evalOneResultAnno(result, anno)

def evalOneBoxResultAnnoCheckExcludePath(result, anno, exclude_path):
    if os.path.exists(exclude_path):
        exclude = cv2.imread(exclude_path, 0)
        return evalOneBoxResultAnnoWithExclude(result, anno, exclude)
    else:
        return evalOneBoxResultAnno(result, anno)

def evalModelResults(results, annos):
    return multiRunStarmap(evalOneResultAnno, results, annos)

def evalModelBoxResults(results, annos):
    return multiRunStarmap(evalOneBoxResultAnno, results, annos)

def evalModelResultsWithExcludePaths(results, annos, exclude_paths):
    return multiRunStarmap(evalOneResultAnnoCheckExcludePath, results, annos, exclude_paths)

def evalModelBoxResultsWithExcludePaths(results, annos, exclude_paths):
    return multiRunStarmap(evalOneBoxResultAnnoCheckExcludePath, results, annos, exclude_paths)

def evalModelResultsWithDir(results_dir, result_suffix, annos_dir, anno_suffix):
    anno_paths = getPathsBySuffix(annos_dir, anno_suffix)
    result_paths = getPathsByTemplate(anno_paths,annos_dir, anno_suffix, results_dir, result_suffix)
    annos = [cv2.imread(anno_path, 0) for anno_path in anno_paths]
    results = [cv2.imread(result_path, 0) for result_path in result_paths]
    assert len(annos) == len(results)
    return evalModelResults(results, annos)

def evalModelBoxResultsWithDir(results_dir, result_suffix, annos_dir, anno_suffix):
    result_paths = getPathsBySuffix(results_dir, result_suffix)
    results = [cv2.imread(result_path, 0) for result_path in result_paths]
    anno_paths = getPathsBySuffix(annos_dir, anno_suffix)   
    annos = [cv2.imread(anno_path, 0) for anno_path in anno_paths]
    return evalModelBoxResults(results, annos)

def getPathsByTemplate(a_paths, a_dir, a_suffix, b_dir, b_suffix):
    res = []
    for path in a_paths:
        res.append(path.replace(a_dir, b_dir).replace(a_suffix, b_suffix))
    return res

def getPathsBySuffix(src_dir, src_suffix):
    paths = sorted(glob.glob(f'{src_dir}/*{src_suffix}'))
    if len(paths)==0:
        paths = sorted(glob.glob(f'{src_dir}/*/*{src_suffix}'))
    #return paths[:10]
    return paths

def getReplacePath(src_path, tar_dir, src_suffix, tar_suffix):
    name = src_path.split('/')[-1].replace(src_suffix, '')
    tar_path = f'{tar_dir}/{name}{tar_suffix}'
    return tar_path

def evalModelResultsWithDirWithExclude(results_dir, result_suffix, annos_dir, anno_suffix, excludes_dir):
    result_paths = getPathsBySuffix(results_dir, result_suffix)
    results = [cv2.imread(result_path, 0) for result_path in result_paths]
    anno_paths = getPathsBySuffix(annos_dir, anno_suffix)
    annos = [cv2.imread(anno_path, 0) for anno_path in anno_paths]
    exclude_paths = []

    anno_suffix = '_tuned.png'
    exclude_suffix = '_excluded.png'
    for anno_path in anno_paths:
        exclude_path = getReplacePath(anno_path, excludes_dir, anno_suffix, exclude_suffix)
        exclude_paths.append(exclude_path)

    return evalModelResultsWithExcludePaths(results, annos, exclude_paths)

def evalModelBoxResultsWithDirWithExclude(results_dir, result_suffix, annos_dir, anno_suffix, excludes_dir):
    result_paths = getPathsBySuffix(results_dir, result_suffix)
    results = [cv2.imread(result_path, 0) for result_path in result_paths]
    anno_paths = getPathsBySuffix(annos_dir, anno_suffix) 
    annos = [cv2.imread(anno_path, 0) for anno_path in anno_paths]
    exclude_paths = []

    anno_suffix = '_tuned.png'
    exclude_suffix = '_excluded.png'
    for anno_path in anno_paths:
        exclude_path = getReplacePath(anno_path, excludes_dir, anno_suffix, exclude_suffix)
        exclude_paths.append(exclude_path)

    return evalModelBoxResultsWithExcludePaths(results, annos, exclude_paths)

def avgArrDictResultsStepThresholdAUC(arr_dict, plot_path):
    _, threshold = arr_dict[0]
    M = []
    keys = arr_dict[0][0][0].keys()
    n = len(arr_dict)
    for i in range(len(threshold)):
        metric = {}
        for k in keys:
            metric[k]=0
        for j in range(n):
            m = arr_dict[j][0][i]
            for k,v in m.items():
                metric[k]+=v
        for k in keys:
            metric[k]=metric[k]/n
        M.append(metric)
    M2 = {}
    for k in keys:
        M2[k]=[]
    for m in M:
        for k in keys:
            M2[k].append(m[k])

    tpr = M2['recall']
    spc = M2['specificity']
    fpr = [1-s for s in spc]
    distance = [math.pow(fpr[i],2)+math.pow((tpr[i]-1),2) for i in range(len(tpr))]
    optimal_idx = np.argmin(np.array(distance))
    optimal_threshold = threshold[optimal_idx]
    metric = M[optimal_idx]
    roc_auc = auc(fpr, tpr)
    metric['auc'] = roc_auc
    metric['threshold'] = optimal_threshold
    drawROC(fpr, tpr, roc_auc, plot_path)
    return metric

def avgArrDictResults(arr_dict):
    a = arr_dict[0]
    metric = {}
    for m in a.keys():
        metric[m]=0
    N = len(arr_dict)
    for a in arr_dict:
        for k,v in a.items():
            metric[k]+=v
    a = arr_dict[0]
    for m in a.keys():
        metric[m]=metric[m]/N
    return metric

if __name__ == '__main__':
    results_dir = '/tmp/modelVSanno/results' 
    annos_dir = '/tmp/modelVSanno/annos' 
    excludes_dir = '/tmp/modelVSanno/excludes'
    result_suffix = '_pred.png'
    anno_suffix = '_tuned.png'
    print(evalModelResultsWithDir(results_dir, result_suffix, annos_dir, anno_suffix))
    print(evalModelBoxResultsWithDir(results_dir, result_suffix, annos_dir, anno_suffix))
    print(evalModelResultsWithDirWithExclude(results_dir, result_suffix, annos_dir, anno_suffix, excludes_dir))
    print(evalModelBoxResultsWithDirWithExclude(results_dir, result_suffix, annos_dir, anno_suffix, excludes_dir))
    print(avgArrDictResults(evalModelBoxResultsWithDirWithExclude(results_dir, result_suffix, annos_dir, anno_suffix, excludes_dir)))

