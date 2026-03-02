# %%
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
from rl_benchmarks.utils.linear_evaluation import get_binary_class_metrics, get_bootstrapped_metrics

from PIL import Image
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
# import umap
import numpy as np
import matplotlib.pyplot as plt
import sklearn

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, cohen_kappa_score, accuracy_score
import torch.nn.functional as F
import sys
import time

import shutil
import os

# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import cv2
import time
from sklearn.preprocessing import label_binarize

import timm
from metrics import report
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm
from rl_benchmarks.metrics import *

from rl_benchmarks.trainers.torch_trainer import TorchTrainer
from rl_benchmarks.models.slide_models.meanpool import MeanPool
from rl_benchmarks.models.slide_models.chowder import Chowder
from rl_benchmarks.models.slide_models.dsmil import DSMIL
from rl_benchmarks.models.slide_models.abmil import ABMIL
from rl_benchmarks.models.slide_models.hiptmil import HIPTMIL
from rl_benchmarks.models.slide_models.transmil import TransMIL

from pathlib import Path
from metrics import report

from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
)

import sys
sys.path.append('/home/yuxin/bme/BCaCAD/model')
from patch_based_test.img import QiLuROI

def get_model(model_name, in_dim, out_dim):
    if model_name == 'abmil':
        model = ABMIL(in_dim, out_dim, d_model_attention=128,temperature= 1.0, mlp_hidden= [128, 64])
    elif model_name == 'chowder':
        model = Chowder(in_dim, out_dim, n_top= 2,n_bottom= 2,tiles_mlp_hidden= [128], mlp_hidden= [128, 64])
    elif model_name == 'dsmil':
        model = DSMIL(in_dim, out_dim, d_tiles_values= 32,d_tiles_queries= 32,passing_values= False,tiles_scores_mlp_hidden=[200,100],
                        tiles_queries_mlp_hidden=[200,100], mlp_hidden=[200,100])
    elif model_name == 'hiptmil':
        model = HIPTMIL(in_dim, out_dim)
    elif model_name == 'transmil':
        model = TransMIL(in_dim, out_features=out_dim)
    elif model_name == 'meanpool':
        model = MeanPool(in_dim, out_dim)
    else:
        raise 'model not found'
    return model


# %%
class IBOTMultiTaskModel(nn.Module):
    def __init__(self, num_classes):
        super(IBOTMultiTaskModel, self).__init__()
        weights_path = '/home/yuxin/Downloads/ibot_vit_base_pancan.pth'
        self.base_model = iBOTViT(architecture="vit_base_pancan", encoder="teacher", weights_path=weights_path)
        # print(self.base_model)
        self.num_features = 768
        self.num_classes = num_classes

        if isinstance(num_classes, list):
            self.heads = nn.ModuleList([nn.Linear(self.num_features, num_class) for num_class in num_classes])
        else:
            self.head = self.base_model.head
            self.head.fc = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        # Forward pass through the base model
        x = self.base_model(x)
        if isinstance(self.num_classes, list):
            x = [head(x) for head in self.heads]
        else:
            x = self.head(x)
        return x

# %%
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
num_classes = [3,3]
img_size = patch_size = 384
data_trans = {
    "train": transforms.Compose([
                                transforms.Resize(img_size),
                                transforms.ColorJitter(),
                                transforms.RandomHorizontalFlip(),
                                # transforms.RandomVerticalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "test": transforms.Compose([
                                transforms.Resize((img_size,img_size)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

# %%

device1 = 'cuda:0'

device2 = 'cuda:1'
device1 = device2 = 'cuda:0'

model = IBOTMultiTaskModel(num_classes)

lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,  # ← use FEATURE_EXTRACTION, not IMAGE_CLASSIFICATION
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.2,
    target_modules=["qkv"],
)
model = get_peft_model(model, lora_config)
model = nn.DataParallel(model)
model.to(device1)

ckpt_path = '/mnt/hd1/bcacad/timm_lora_ft/2025_07_14_04_18_54/model-13.pth'
ckpt = torch.load(ckpt_path, map_location=device1)
model.load_state_dict(ckpt['model_state_dict'])
feature_extractor = model.module.base_model.base_model.eval()


# %%
class Feature_Fly(Dataset):
    def get_label(self, type):
        D = {
            'normal':[0,0],
            'dcis-1':[1,0],
            'dcis-2':[1,1],
            'dcis-3':[1,2],
            'ibc-1':[2,0],
            'ibc-2':[2,1],
            'ibc-3':[2,2],
        }
        label =  D[type]
        return label[0],label[1]
    
    def __init__(self, image_dir, phase='test', train_tile=None):
        folder = Path(image_dir)
        # self.im_paths = list(folder.rglob('**/*.*'))
        # self.labels = [self.get_label(path.parent.name) for path in self.im_paths]
        self.wsi_dirs = list(folder.glob(f'*/*'))
        self.phase=phase
        self.train_tile = train_tile
        self.size = 336
        self.bs = 16
        self.src_mag = 10
        self.tar_mag = 10
    
        
    def __len__(self):
        return len(self.wsi_dirs)
    def __getitem__(self, item):
        wsi_dir = self.wsi_dirs[item]
        im_paths = list(wsi_dir.rglob('**/*.png'))
        wsi_patches = []
        for im_path in im_paths:
            im = QiLuROI(str(im_path), self.src_mag, self.tar_mag, self.size)
            im.setIterator(self.size)
            patches = [p for p in im]
            wsi_patches.extend(patches)
        patches = wsi_patches
        if self.phase == 'train':
            indices = np.random.choice(len(patches), self.train_tile, replace=True)
            patches = [patches[i] for i in indices]
        patches = [data_trans[self.phase](p) for p in patches]
        bs = self.bs
        for i in range(0, len(patches), bs):
            x = torch.stack(patches[i:i+bs], dim=0)
            x = x.to(device1)
            y = feature_extractor(x)
            if i == 0:
                features = y.detach().cpu().numpy()
            else:
                # features = torch.concatenate([features, y], dim=0)
                np.concatenate([features, y.detach().cpu().numpy()], axis=0)
        label = self.get_label(wsi_dir.parent.name)
        return features, label
    
    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images,  labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        # masks = torch.as_tensor(masks)

        return images, labels

class Feature(Dataset):
    def get_label(self, type):
        D = {
            'normal':[0,0],
            'dcis-1':[1,0],
            'dcis-2':[1,1],
            'dcis-3':[1,2],
            'ibc-1':[2,0],
            'ibc-2':[2,1],
            'ibc-3':[2,2],
        }
        label =  D[type]
        return label[0],label[1]
    
    def __init__(self, feature_dir, phase='test'):
        folder = Path(feature_dir)
        feature_paths = list(folder.rglob('**/*.npy'))
        self.labels = [self.get_label(path.parent.name) for path in feature_paths]
        self.features = [np.load(path) for path in feature_paths]
        self.phase=phase
        
    def __len__(self):
        return len(self.features)
    def __getitem__(self, item):
        features = self.features[item]
        if self.phase == 'train':
            indices = np.random.choice(features.shape[0], 8, replace=True)
            features = np.stack([features[i] for i in indices], axis=0)
        return features, self.labels[item]
    
    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images,  labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        # masks = torch.as_tensor(masks)

        return images, labels


# %%
cfg = dict(
    epoch=10,
    bs=8,
    train_tiles = 32,
)

# %%
data_root = Path('/mnt/hd0/project/bcacad/data/wsi-level')
train_set = Feature_Fly(data_root / 'suqh' / 'model', 'train', cfg['train_tiles'])

# %%
criterion = nn.CrossEntropyLoss()
merics = {'acc': compute_multiclass_accuracy, 'auc': compute_mean_one_vs_all_auc}
input_dim = 768
out_dim = [3,3]
tasks = ['type', 'nonibc', 'ibc']
class_names = {
    'type': ['Normal', 'nonIBC', 'IBC'],
    'nonibc': ['Low', 'Medium', 'High'],
    'ibc': ['Low', 'Medium', 'High'],
}

# %%
torch.cuda.device_count()

# %%
# model_names = [ 'abmil', 'meanpool','chowder', 'transmil', 'dsmil', 'hiptmil']
model_names = ['abmil']
# device = 'cuda'
models={}
for model_name in model_names:
    # print(model_name)
    model = get_model(model_name, input_dim, out_dim).to('cuda')
    model = nn.DataParallel(model).to(device1)
    trainer = TorchTrainer(model, criterion, merics, device='cuda', num_epochs=cfg['epoch'], batch_size=cfg['bs'], learning_rate=1e-4)
    res = trainer.train(train_set, train_set)
    # res = trainer.train(fake, fake)
    models[model_name] = model


    pha = 'train'
    labels = np.array(res[pha][0])
    probs = np.array(res[pha][1])
    preds = np.array(res[pha][2])

    type_labels = labels[0]
    type_probs = probs[0]
    type_preds = preds[0]

    nonibc_index = np.where(type_labels ==1)
    nonibc_labels = labels[1][nonibc_index]
    nonibc_probs = probs[1][nonibc_index]
    nonibc_preds = preds[1][nonibc_index]

    ibc_index = np.where(type_labels ==2)
    ibc_labels = labels[1][ibc_index]
    ibc_probs = probs[1][ibc_index]
    ibc_preds = preds[1][ibc_index]

    re = {}
    avg_aucs = {}
    re['type'] = report(type_labels, type_preds, type_probs, class_names['type'])
    avg_aucs['type'] = compute_mean_one_vs_all_auc(type_labels, type_probs)

    re['nonibc'] = report(nonibc_labels, nonibc_preds, nonibc_probs, class_names['nonibc'])
    avg_aucs['nonibc'] = compute_mean_one_vs_all_auc(nonibc_labels, nonibc_probs)

    re['ibc'] = report(ibc_labels, ibc_preds, ibc_probs, class_names['ibc'])
    avg_aucs['ibc'] = compute_mean_one_vs_all_auc(ibc_labels, ibc_probs)

    for task in ['type', 'nonibc', 'ibc']:
        r = re[task]
        fs = "{} {} {} acc: {:.4f}, auc: {:.4f} [{:.4f} {:.4f} {:.4f}]".format(model_name, pha, task, r['accuracy'], avg_aucs[task], r['0']['auc'], r['1']['auc'], r['2']['auc'])
        print(fs)
    print()
print()
    
save_root = Path('/mnt/hd0/project/bcacad/model/wsi_models/model_lora_ft_2')
if not save_root.exists():
    save_root.mkdir()
for model_name in model_names:
    torch.save(models[model_name].state_dict(), save_root / f'{model_name}.pth')
    print(f'save {model_name}.pth')
    


# %%
class WSI(Dataset):
    def get_label(self, type):
        D = {
            'normal':[0,0],
            'dcis-1':[1,0],
            'dcis-2':[1,1],
            'dcis-3':[1,2],
            'ibc-1':[2,0],
            'ibc-2':[2,1],
            'ibc-3':[2,2],
        }
        label =  D[type]
        return label[0],label[1]
    
    def __init__(self, feature_dir, phase='test'):
        folder = Path(feature_dir)
        wsi_dirs = list(folder.glob('*/*'))
        # self.labels = [self.get_label(path.parent.name) for path in wsi_dirs]
        self.labels = []
        for path in wsi_dirs:
            label = path.parent.name
            cvtlabel = self.get_label(label)
            self.labels.append(cvtlabel)
        # self.features = [np.load(path) for path in feature_paths]
        self.features = []
        for wsi_dir in wsi_dirs:
            feature_paths = list(wsi_dir.rglob('**/*.npy'))
            features = []
            for path in feature_paths:
                patch_features = torch.from_numpy(np.load(path)).to(device1).unsqueeze(0)
                # roi_feats = feature_model.extract_features(patch_features)
                features.extend(patch_features)
            self.features.append(features)
        self.wsis = wsi_dirs
        self.phase=phase
        
    def __len__(self):
        return len(self.wsis)
    def __getitem__(self, item):
        features = self.features[item]
        # wsi = self.wsis[item]
        if self.phase == 'train':
            indices = np.random.choice(len(features), 6, replace=True)
            features = torch.cat([features[i] for i in indices], dim=0)
            return features, self.labels[item]
        features = torch.cat(features, dim=0)
        return features, self.labels[item]
    
    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images,  labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        # masks = torch.as_tensor(masks)

        return images, labels

# %%


# %%
# model_names = [ 'abmil']
# models={}
# device = 'cuda'
# model_dir = Path('/mnt/hd0/project/bcacad/model/wsi_models/model2')
# for model_name in model_names:
#     model = get_model(model_name, input_dim, out_dim).to(device)
#     model = nn.DataParallel(model, device_ids=[0,1], output_device=0)
#     model.load_state_dict(torch.load(model_dir / f'{model_name}.pth'))
#     models[model_name] = model.to(device)

# %%
# testing
pha='test'
feature_root = Path('/mnt/hd0/project/bcacad/model/wsi_features_lora_ft')
test_cohorts = ['suqh_full', 'qduh', 'shsu','bjszhp','bracs', 'bcnb', 'aggregate']
# test_cohorts = [ 'bcnb', 'bach', 'apght']
# model_names = [ 'abmil', 'meanpool','chowder', 'transmil']
model_names = ['abmil']
# load_dir = Path('/mnt/hd0/project/bcacad/model/roi_models/model2')
for cohort in test_cohorts:
    for model_name in model_names:
        # model = get_model(model_name, input_dim, out_dim).to(device)
        model = models[model_name]
        # ckpt = torch.load(load_dir / f'{model_name}.pth')
        # model.load_state_dict(ckpt)
        # model = model.to(device)

        test_set = WSI(feature_root / cohort / 'test', 'test')
        print(cohort, len(test_set))
        trainer = TorchTrainer(model, criterion, merics, device=device1)
        res = trainer.predict(test_set)

        labels = np.array(res[0])
        probs = np.array(res[1])
        preds = np.array(res[2])

        type_labels = labels[0]
        type_probs = probs[0]
        type_preds = preds[0]

        nonibc_index = np.where(type_labels ==1)
        nonibc_labels = labels[1][nonibc_index]
        nonibc_probs = probs[1][nonibc_index]
        nonibc_preds = preds[1][nonibc_index]

        ibc_index = np.where(type_labels ==2)
        ibc_labels = labels[1][ibc_index]
        ibc_probs = probs[1][ibc_index]
        ibc_preds = preds[1][ibc_index]

        re = {}
        avg_aucs = {}
        if type_labels.size == 0:
            re['type'] = None
            avg_aucs['type'] = None
        else:
            re['type'] = report(type_labels, type_preds, type_probs, class_names['type'])
            avg_aucs['type'] = compute_mean_one_vs_all_auc(type_labels, type_probs)

        if nonibc_labels.size == 0:
            re['nonibc'] = None
            avg_aucs['nonibc'] = None
        else:
            re['nonibc'] = report(nonibc_labels, nonibc_preds, nonibc_probs, class_names['nonibc'])
            avg_aucs['nonibc'] = compute_mean_one_vs_all_auc(nonibc_labels, nonibc_probs)

        if ibc_labels.size == 0:
            re['ibc'] = None
            avg_aucs['ibc'] = None
        else:
            re['ibc'] = report(ibc_labels, ibc_preds, ibc_probs, class_names['ibc'])
            avg_aucs['ibc'] = compute_mean_one_vs_all_auc(ibc_labels, ibc_probs)

        for task in ['type', 'nonibc', 'ibc']:
            r = re[task]
            if r is None:
                continue
            fs = "{} {} {} acc: {:.4f}, auc: {:.4f} [{:.4f} {:.4f} {:.4f}]".format(cohort, model_name, task, r['accuracy'], avg_aucs[task], r['0']['auc'], r['1']['auc'], r['2']['auc'])
            print(fs)
        print()

# %%


# %%



