#!/usr/bin/env python
# coding: utf-8

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
from rl_benchmarks.metrics import *
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
from metrics import report

# from apex import amp


def copy_self(destination):
    # 获取当前脚本的绝对路径
    script_path = os.path.realpath(__file__)
    
    # 拷贝文件到目标位置
    shutil.copy2(script_path, destination)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
num_classes = [3,3]
img_size = patch_size = 384

data_trans = {
    "train": transforms.Compose([
                                transforms.Resize(img_size),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "test": transforms.Compose([
                                # transforms.Resize(img_size),
                                transforms.CenterCrop(336),
                                transforms.Resize(img_size),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

class MyDataSet(Dataset):
    """自定义数据集"""

    # def __init__(self, images_path: list, images_class: list, images_ncl: list, images_epi: list, images_tub: list, images_mit: list, transform=None):
    
    def get_type_grade(self, path):
        type_grade = path.parent.name
        if type_grade == 'normal':
            return 0,0
        elif type_grade == 'dcis-1':
            return 1,0
        elif type_grade == 'dcis-2':
            return 1,1
        elif type_grade == 'dcis-3':
            return 1,2
        elif type_grade == 'ibc-1':
            return 2,0
        elif type_grade == 'ibc-2':
            return 2,1
        elif type_grade == 'ibc-3':
            return 2,2
    
    def __init__(self, data_root:pathlib, transform=None):
        self.images_path = list((data_root).glob('**/*.*'))
        self.images_class = [self.get_type_grade(path) for path in self.images_path]
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        # img_full, img_name = os.path.split(self.images_path[item])
        # img_n, ext = os.path.splitext(img_name)
        min_size = 336
        min_size = int(min_size)
        # img = Image.open(self.path)
        w ,h = img.size

        if w < min_size or h < min_size:
            new_h = min_size if h < min_size else h
            new_w = min_size if w < min_size else w
            pad_tran = transforms.RandomCrop((new_h,new_w), padding_mode='reflect', pad_if_needed=True)
            img = pad_tran(img)

        # seed = np.random.randint(2147483647)
        if self.transform is not None:
            # torch.manual_seed(seed)
            # torch.cuda.manual_seed(seed)
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)

        return images, labels


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
    
class TransMultiTaskModel(nn.Module):
    def __init__(self, base_model_name, num_classes):
        super(TransMultiTaskModel, self).__init__()
        ckpt_dict = {
            'swin_base_patch4_window12_384_in22k': dict(file='/home/yuxin/Downloads/swin_base_patch4_window12_384_22k.pth'),
            'swin_base_patch4_window7_224_in22k': dict(file='/home/yuxin/Downloads/swin_base_patch4_window7_224_22k.pth'),
            'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k': dict(file='/home/yuxin/Downloads/swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth'),
            'convnext_large.fb_in22k_ft_in1k_384': dict(file='/home/dc/yuxinwu/breast_cad/convnext_large_22k_1k_384.pth'),
            'convnext_base.fb_in22k_ft_in1k_384': dict(file='/home/dc/yuxinwu/breast_cad/convnext_base_22k_1k_384.pth'),
            'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k': dict(file='/home/yuxin/Downloads/swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth'),
        }
        self.base_model = timm.create_model(base_model_name, pretrained=True, pretrained_cfg_overlay=ckpt_dict[base_model_name])
        # print(self.base_model)
        self.num_features = self.base_model.num_features
        self.num_classes = num_classes

        self.base_model.feature = nn.Sequential(
            self.base_model.head.global_pool,
            self.base_model.head.drop,
        )
        if isinstance(num_classes, list):
            self.heads = nn.ModuleList([nn.Linear(self.num_features, num_class) for num_class in num_classes])
        else:
            self.head = self.base_model.head
            self.head.fc = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        # Forward pass through the base model
        x = self.base_model.forward_features(x)
        # print(x.shape)
        x = self.base_model.feature(x)
        # print(x.shape)

        if isinstance(self.num_classes, list):
            x = [head(x) for head in self.heads]
        else:
            x = self.head(x)
        return x

model = IBOTMultiTaskModel(num_classes)

# MultiTaskModel = TransMultiTaskModel
# model_name = 'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k'
# model = MultiTaskModel(model_name, num_classes)
device = "cuda"
model.to(device)

# model = nn.DataParallel(model)

def iterate(model, data_loader, device, optimizer=None, phase='test', epoch='Test'):
    if phase == 'train':
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()
    
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    sample_num = 0
    tasks = ['type', 'grade']
    y_true, y_pred, y_prob = {}, {}, {}
    for t in tasks:
        y_true[t], y_pred[t], y_prob[t] = [], [], []
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        sample_num += images.shape[0]

        outputs = model.forward(images)

        losses = []

        for i in range(len(outputs)):
            task = tasks[i]
            output = outputs[i]
            pred = torch.max(output, dim=1)[1]
            label = labels[:,i]
            # print(output.shape, label.shape)
            loss = loss_function(output, label)
            losses.append(loss)


            y_pred[task].extend(pred.tolist())
            y_true[task].extend(label.to(device).tolist())
            
            prob = F.softmax(output, dim=1)
            y_prob[task].extend(prob.to(device).tolist())
        
        loss = sum(losses) / len(losses)
        
        accu_loss += loss.detach()

        # compute acc
        acc = []
        f1_macro = []
        for task in tasks:
            acc.append(accuracy_score(y_true[task], y_pred[task]))
            f1_macro.append(f1_score(y_true[task], y_pred[task], average='macro'))


        data_loader.desc = "[{} {}] loss: {:.3f}, acc: {:.3f},{:.3f}, f1_marco: {:.3f},{:.3f}".format(
            phase, epoch, accu_loss.item() / (step + 1), acc[0], acc[1], f1_macro[0], f1_macro[1])

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        if phase == 'train':
            loss.backward()
            optimizer.first_step(zero_grad=True)
            outputs = model.forward(images)
            losses = []
            for i in range(len(outputs)):
                output = outputs[i]
                loss = loss_function(output, labels[:,i])
                losses.append(loss)
            loss = sum(losses) / len(losses)
            loss.backward()
            optimizer.second_step(zero_grad=True)
            scheduler.step()

    return accu_loss.item() / (step + 1), acc, f1_macro, y_true, y_pred, y_prob

model = nn.DataParallel(model)
# model = nn.DataParallel(model)

# ckpt_path = '/mnt/hd0/project_large_files/bca_grading/suqh/swin_multi_task/2023_12_15_18:17:54/model-34.pth'
# ckpt_path = '/mnt/hd0/project_large_files/bca_grading/suqh/swin_multi_task/2024_01_02_17_39_16/model-9.pth'

# ckpt = torch.load(ckpt_path, map_location=device)
# model.load_state_dict(ckpt['model_state_dict'])
# weights_dir = pathlib.Path('/mnt/hd0/project_large_files/bca_grading/suqh/swin_multi_task/2023_12_22_10_43_44')
# weights_dir = pathlib.Path('/mnt/hd0/project_large_files/bca_grading/suqh/swin_multi_task/2024_01_03_09_24_37')
weights_dir = pathlib.Path('/mnt/hd0/project/bcacad/model/pretrainSSL_ibot_vit+ibot_ft+fsl_ft')
logs_save_path = weights_dir
if os.path.exists(logs_save_path) is False:
    os.makedirs(logs_save_path)
log_filename = 'independent_test_'+time.strftime("%Y_%m_%d_%H_%M_%S") + '.txt'
log_filepath = os.path.join(logs_save_path, log_filename)
weight_path = '/mnt/hd0/project/bcacad/model/pretrainSSL_ibot_vit+ibot_ft+fsl_ft/model-5.pth'
copy_self(weights_dir)
model.load_state_dict(torch.load(weight_path, map_location=device)['model_state_dict'])

bs=16
# 实例化训练数据集

class_names = {
    'type': ['Normal', 'nonIBC', 'IBC'],
    'grade': ['Low', 'Medium', 'High']
}
# 
cohorts = ['suqh_full', 'qduh', 'shsu', 'bracs', 'bcnb', 'bach', 'apght']


# data_root = pathlib.Path('/mnt/hd0/project/bcacad/data/patch-level')
data_root = pathlib.Path('/mnt/hd0/project/bcacad/data/roi-level')

datasets = {}
for cohort in cohorts:
    datasets[cohort] = MyDataSet(data_root / cohort / 'test', data_trans['test'])
dataloaders = {}
for cohort in cohorts:
    dataloaders[cohort] = torch.utils.data.DataLoader(datasets[cohort],
                                            batch_size=bs,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=0,
                                            collate_fn=datasets[cohort].collate_fn)

for cohort in cohorts:
    print(cohort)

    # validate
    _,_,_,y_true,y_pred,y_prob  = iterate(model=model,
                data_loader=dataloaders[cohort],
                device=device,
                epoch=cohort)

    # class_level
    Acc = {}
    F1score = {}
    Auc = {}
    Kappa = {}
    Precision = {}
    Recall = {}
    # for task in ['type', 'grade']:
        
    #     Acc[task] = accuracy_score(y_true[task], y_pred[task])
    #     y_true_b = label_binarize(y_true[task], classes=[0, 1, 2])
    #     Auc[task] = roc_auc_score(y_true_b, y_prob[task], multi_class='ovo', average=None)
    #     Kappa[task] = cohen_kappa_score(y_true[task], y_pred[task], weights='quadratic')
    #     Precision[task], Recall[task], F1score[task], _ = precision_recall_fscore_support(y_true[task], y_pred[task])
    
    # for type
    if cohort in ['suqh_full', 'qduh', 'shsu', 'bracs', 'bach']:
        type_y_pred = y_pred['type']
        type_y_prob = y_prob['type']
        type_y_true = y_true['type']
        if cohort in ['shsu']:
            _y_preds = []
            _y_trues = []
            _y_probs = []
            for i in range(len(type_y_pred)):
                _y_pred = type_y_pred[i]
                _y_true = type_y_true[i]
                _y_prob = type_y_prob[i]
                if _y_pred > 0:
                    _y_pred -= 1
                if _y_true > 0:
                    _y_true -= 1
                _y_prob = [_y_prob[0]+_y_prob[1], _y_prob[2]]
                _y_preds.append(_y_pred)
                _y_trues.append(_y_true)
                _y_probs.append(_y_prob)
            type_y_pred = np.array(_y_preds)
            type_y_true = np.array(_y_trues)
            type_y_prob = np.array(_y_probs)
            type_y_true_b = [[1 if i == j else 0 for j in range(2)] for i in type_y_true]
            Auc['type'] = roc_auc_score(type_y_true_b, type_y_prob, multi_class='ovo', average=None)
            classes = class_names['type'][1:]
        else:
            type_y_true_b = label_binarize(type_y_true, classes=[0, 1, 2])
            Auc['type'] = roc_auc_score(type_y_true_b, type_y_prob, multi_class='ovo', average=None)
            classes = class_names['type']
        Acc['type'] = accuracy_score(type_y_true, type_y_pred)
        
        Kappa['type'] = cohen_kappa_score(type_y_true, type_y_pred, weights='quadratic')
        Precision['type'], Recall['type'], F1score['type'], _ = precision_recall_fscore_support(type_y_true, type_y_pred)
        log_txt_formatter = "\n{cohort}: \n" \
            "[Type] [AUC] {auc_t} [ACC] {acc_t} [F1-Score] {f1_t} [Kappa] {kappt_t} [Precision] {p_t} [Recall] {r_t}\n\n"
        to_write = log_txt_formatter.format(cohort="{: <12}".format(cohort),
                                            acc_t = Acc['type'],
                                            f1_t = F1score['type'],
                                            auc_t = Auc['type'],
                                            kappt_t = Kappa['type'],
                                            p_t = Precision['type'],
                                            r_t = Recall['type'],
                                        )
        r = report(type_y_true, type_y_pred, type_y_prob, classes)
        to_write = to_write + '\n' + str(r)
        print(to_write)
        with open(log_filepath, "a") as f:
            f.write(to_write)

    # for grade noIBC
    if cohort in ['suqh_full', 'qduh', 'shsu']:
        tumor_idx = np.where(np.array(y_true['type']) == 1)[0]
        grade_y_pred = np.array(y_pred['grade'])[tumor_idx]
        grade_y_prob = np.array(y_prob['grade'])[tumor_idx]
        grade_y_true = np.array(y_true['grade'])[tumor_idx]
        Acc['grade'] = accuracy_score(grade_y_true, grade_y_pred)
        grade_y_true_b = label_binarize(grade_y_true, classes=[0, 1, 2])
        Auc['grade'] = roc_auc_score(grade_y_true_b, grade_y_prob, multi_class='ovo', average=None)
        Kappa['grade'] = cohen_kappa_score(grade_y_true, grade_y_pred, weights='quadratic')
        Precision['grade'], Recall['grade'], F1score['grade'], _ = precision_recall_fscore_support(grade_y_true, grade_y_pred)

        log_txt_formatter = "\n{cohort}: \n" \
            "[Grade-nonIBC] [AUC] {auc_g} [ACC] {acc_g} [F1-Score] {f1_g} [Kappa] {kappt_g} [Precision] {p_g} [Recall] {r_g}\n\n"
        to_write = log_txt_formatter.format(cohort="{: <12}".format(cohort),
                                            acc_g = Acc['grade'],
                                            f1_g = F1score['grade'],
                                            auc_g = Auc['grade'],
                                            kappt_g = Kappa['grade'],
                                            p_g = Precision['grade'],
                                            r_g = Recall['grade'],    
                                        )
        
        r = report(grade_y_true, grade_y_pred, grade_y_prob, class_names['grade'])
        to_write = to_write + '\n' + str(r)
        print(to_write)
        with open(log_filepath, "a") as f:
            f.write(to_write)

    # for grade IBC
    if cohort in ['suqh_full', 'qduh', 'shsu', 'apght', 'bcnb']:
        tumor_idx = np.where(np.array(y_true['type']) == 2)[0]
        grade_y_pred = np.array(y_pred['grade'])[tumor_idx]
        grade_y_prob = np.array(y_prob['grade'])[tumor_idx]
        grade_y_true = np.array(y_true['grade'])[tumor_idx]
        Acc['grade'] = accuracy_score(grade_y_true, grade_y_pred)
        grade_y_true_b = label_binarize(grade_y_true, classes=[0, 1, 2])
        Auc['grade'] = roc_auc_score(grade_y_true_b, grade_y_prob, multi_class='ovo', average=None)
        Kappa['grade'] = cohen_kappa_score(grade_y_true, grade_y_pred, weights='quadratic')
        Precision['grade'], Recall['grade'], F1score['grade'], _ = precision_recall_fscore_support(grade_y_true, grade_y_pred)

        log_txt_formatter = "\n{cohort}: \n" \
            "[Grade-IBC] [AUC] {auc_g} [ACC] {acc_g} [F1-Score] {f1_g} [Kappa] {kappt_g} [Precision] {p_g} [Recall] {r_g}\n\n"
        to_write = log_txt_formatter.format(cohort="{: <12}".format(cohort),
                                            acc_g = Acc['grade'],
                                            f1_g = F1score['grade'],
                                            auc_g = Auc['grade'],
                                            kappt_g = Kappa['grade'],
                                            p_g = Precision['grade'],
                                            r_g = Recall['grade'],    
                                        )
        r = report(grade_y_true, grade_y_pred, grade_y_prob, class_names['grade'])
        to_write = to_write + '\n' + str(r)
        print(to_write)
        with open(log_filepath, "a") as f:
            f.write(to_write)