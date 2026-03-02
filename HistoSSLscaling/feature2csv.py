from PIL import Image
from rl_benchmarks.models import iBOTViT
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
import pathlib
from tqdm import tqdm
from PIL import Image
import numpy as np
Image.MAX_IMAGE_PIXELS = None
from torchvision import transforms
import torch
from torch.utils.data import Dataset
import os
from multiprocessing import Pool
import umap
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from rl_benchmarks.utils.linear_evaluation import get_binary_class_metrics, get_bootstrapped_metrics, dict_to_dataframe
from sklearn.metrics import f1_score,precision_recall_fscore_support,cohen_kappa_score,roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix

import pandas as pd




patch_size = 224
data_root = pathlib.Path('/mnt/hd0/original_datasets/JT_Breast/SUQH/40xTiles')
split_root = pathlib.Path('/mnt/hd0/project_large_files/bca_grading/suqh/split_data')
feature_root = pathlib.Path('/mnt/hd0/project_large_files/bca_grading/suqh/histosslscaling_finetune/full_2023_10_24_10:42:58')
phases = ['train', 'val', 'test']
# phases = ['test']
save_root = feature_root / f'center_crop_features_10x_{patch_size}'

features = {}
labels = {}
names = {}

for phase in phases:
    phase_root = save_root / phase
    feature_paths = list(phase_root.glob('**/*.npy'))
    _labels = [int(path.stem.split('-')[-1])-1 for path in feature_paths]
    _labels = [1 if label==1 else 0 for label in _labels] # grade 1,3 vs grade 2
    labels[phase] = _labels
    features[phase] = [np.load(path) for path in feature_paths]
    names[phase] = [path.stem for path in feature_paths]

# save to csv

dfs = {}
for phase in phases:
    data = {}
    data['label'] = labels[phase]
    # this patient_id is not real patient id, its just the name of the image, for use of pypathomics
    data['patient_id'] = names[phase]
    for i in range(len(features[phase][0])):
        data[f'feature{i}'] = [feature[i] for feature in features[phase]]
    dfs[phase] = pd.DataFrame(data=data)

# concat train and val
df =  pd.concat([dfs['train'], dfs['val']])
df.to_csv(feature_root / 'train_val_feature.csv', index=None)

dfs['test'].to_csv(feature_root / 'test_feature.csv', index=None)









"""

    # binary_metrics = get_binary_class_metrics(labels, scores)

    # bootstrapped_metrics= get_bootstrapped_metrics(labels, scores, n_resamples, confidence_level)

    # results_dict = {
    #     "binary": {'default': binary_metrics},
    #     "bootstrap": {'default': bootstrapped_metrics},
    # }

    # print(results_dict)

    # results = dict_to_dataframe(
    #     results_dict, metrics=["auc", "acc", "f1"], class_names=['Grade 1', 'Grade 2', 'Grade 3']
    # )

    # results.to_excel(f'{phase}.xlsx', index=None)
    y_pred = preds
    y_true = labels

    f1_macro=f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')

    qwk = cohen_kappa_score(y_true, y_pred, labels=None, weights='quadratic', sample_weight=None)

    y_true2 = label_binarize(y_true, classes=[0, 1, 2])
    auc = roc_auc_score(y_true2, scores, multi_class='ovo', average='macro')
    auc_class = roc_auc_score(y_true2, scores, multi_class="ovo", average=None)

    test_auc = auc
    test_acc = f1_micro

    # get confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(cm)


    p_class, r_class, f_class, support_micro = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred,
                                                                               average=None)

    log_txt_formatter = "{phase}: [AUC] {test_auc} [ACC] {test_acc} [f1_macro] {test_f1_ma} [Q-wK] {qwk}" \
                            " [AUC] {auc_c} [P] {p_c} [R] {r_c} [F] {f_c}\n"

    to_write = log_txt_formatter.format(phase="{: <4}".format(phase),
                                        test_auc=" ".join(["{}".format('%.3f' % test_auc)]),
                                        test_acc=" ".join(["{}".format('%.3f' % test_acc)]),
                                        test_f1_ma=" ".join(["{}".format('%.3f' % f1_macro)]),
                                        qwk=" ".join(["{}".format('%.3f' % qwk)]),
                                        auc_c=auc_class,
                                        p_c=p_class,
                                        r_c=r_class,
                                        f_c=f_class,)
    print(to_write) 



"""


