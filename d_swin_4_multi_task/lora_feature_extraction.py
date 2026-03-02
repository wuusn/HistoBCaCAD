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

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
import torch.nn.functional as F
import sys
import time
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
from sam import SAM
# from apex import amp
import sys
sys.path.append('/home/yuxin/bme/BCaCAD/model')
from patch_based_test.img import QiLuROI
# from model import swin_tiny_patch4_window7_224 as create_model
# from model import swin_base_patch4_window7_224 as create_model

# lora part
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

img_size = patch_size = 384

num_classes = [3,3]

to_tensor= transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class IBOTMultiTaskModel(nn.Module):
    def __init__(self, num_classes):
        super(IBOTMultiTaskModel, self).__init__()
        weights_path = '//home/yuxin/Downloads/ibot_vit_base_pancan.pth'
        self.base_model = iBOTViT(architecture="vit_base_pancan", encoder="teacher", weights_path=weights_path)

        # freeze
        # Freeze all layers:
        for param in self.base_model.parameters():
            param.requires_grad = False
        # print(self.base_model)
        self.num_features = 768
        self.num_classes = num_classes

        if isinstance(num_classes, list):
            self.heads = nn.ModuleList([nn.Linear(self.num_features, num_class) for num_class in num_classes])
        else:
            self.head = self.base_model.head
            self.head.fc = nn.Linear(self.num_features, num_classes)

    def forward(self, x, **kwargs):
        # Forward pass through the base model
        x = self.base_model(x)
        if isinstance(self.num_classes, list):
            x = [head(x) for head in self.heads]
        else:
            x = self.head(x)
        return x

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
model.to(device)

ckpt_path = '/mnt/hd1/bcacad/timm_lora_ft/2025_07_14_04_18_54/model-13.pth'
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])

feature_extractor = model.module.base_model.base_model.eval()

# rand_input = torch.rand(10,3,336,336).to(device)
# print(feature_extractor(rand_input).shape)

from pathlib import Path
data_root = Path('/mnt/hd0/project/bcacad/data/roi-level')
save_root = Path('/mnt/hd0/project/bcacad/model/roi_features_ft_lora')
test_cohorts = ['bjszhp']
# test_cohorts = ['qduh', 'shsu', 'suqh_all_patch','bracs', 'bcnb', 'bach', 'apght',]
# test_cohorts = ['suqh_full']

def to_feature(model, device, path):
    size = 336
    bs = 8
    im = QiLuROI(str(path), 10, 10, size)
    im.setIterator(size)
    patches = [to_tensor(p) for p in im]
    for i in range(0, len(patches), bs):
        x = torch.stack(patches[i:i+bs], dim=0)
        x = x.to(device)
        y = model(x)
        if i == 0:
            features = y.detach().cpu().numpy()
        else:
            features = np.concatenate([features, y.detach().cpu().numpy()], axis=0)
    return features

for cohort in test_cohorts:
    print(cohort)
    if cohort == 'bjszhp':
        src_dir = data_root / cohort
    else:
        src_dir = data_root / cohort / 'test'
    save_dir = save_root / cohort
    im_files = list(src_dir.glob('**/*.*'))
    for im_file in im_files:
        feature = to_feature(feature_extractor, device, im_file)
        if cohort == 'bjszhp':
            label = im_file.parent.name
            save_path = save_dir / 'test' / label /f'{im_file.stem}.npy'
        else:
            save_path = save_dir / 'test' /f'{im_file.stem}.npy'

        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, feature)
            