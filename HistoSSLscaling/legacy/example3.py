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

from Bclassifiers import QDA_, LDA_, RandomForestC_, SGDClassifier_, DecisionTree_, \
    KNeighbors_, LinearSVC_, SVC_rbf_, GaussianProcess_, \
    MLPClassifier_, AdaBoostClassifier_, GaussianNB_, Ridge_

from Bfeatures_selection import UnivariateFeatureSelection, mutual_info_selection, mrmr_selection, \
    ttest_selection, ranksums_selection, varianceThreshold_rm

from sklearn.feature_selection import VarianceThreshold


def findIndexOfFeatures(fullFeatures, selectedFeatrues):
    ind_list = [list(fullFeatures).index(selectedFeatrues[i]) for i in range(len(selectedFeatrues))]
    return ind_list

list_classifers_Dict = {'QDA': QDA_,
                        'LDA': LDA_,
                        'RandomForest': RandomForestC_,
                        'DecisionTree': DecisionTree_,
                        'KNeigh': KNeighbors_,
                        # 'LinearSVC': LinearSVC_,
                        # 'MLP': MLPClassifier_,
                        # 'GaussianNB': GaussianNB_,
                        # 'SGD': SGDClassifier_,
                        # 'SVC_rbf': SVC_rbf_,
                        # 'AdaBoost': AdaBoostClassifier_
                        }  ###Ridge_, GaussianProcess_

list_feats_selection_Dict = {'Univariate': UnivariateFeatureSelection,
                            #  'mutualInfo': mutual_info_selection,
                             'mrmr': mrmr_selection,
                             'ttest': ttest_selection,
                             'ranksums': ranksums_selection}


train_val_path = 'train_val_feature.csv'
test_path = 'test_feature.csv'
corr_threshold = .9

df_features_labels = pd.read_csv(train_val_path)
df_features_labels = df_features_labels.reset_index(drop=True)
df_labels = df_features_labels[['label']]
df_patients_id = df_features_labels[['patient_id']]
df_features = df_features_labels.drop(columns=['patient_id', 
                                                   'label'])
features_name = df_features.columns.to_list()


best_auc = -1
best_classifier_name = None
best_selection = None
n_features = 100
best_model = None

X = df_features
# X = df_features.to_numpy()
# X = pd.DataFrame(X, columns=features_name)
y = df_labels.to_numpy()

# find best model
for classifier_name, classifier in list_classifers_Dict.items():
    for slection_name, selection in list_feats_selection_Dict.items():
        _, selected_features = selection(X=X, y=y, n_features=n_features)
        ind_selected_features = findIndexOfFeatures(fullFeatures=X,
                                                    selectedFeatrues=selected_features)

        y_pred, y_score_pred, best_params = classifier(selected_features, y).search_params(selected_features)
        auc = roc_auc_score(y, y_score_pred[:,1])
        if auc > best_auc:
            best_auc = auc
            best_classifier_name = classifier_name
            best_selection = slection_name
            best_params = best_params
            best_model = classifier(selected_features, y).train(selected_features, y, best_params)

print('best_auc: ', best_auc)
# train best with all train and val data



