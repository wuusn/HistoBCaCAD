#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pathlib
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score
from tqdm import tqdm
import sys
import time
import os
from rl_benchmarks.models import iBOTViT
from rl_benchmarks.metrics import *
import shutil
import argparse
import timm
from torchvision.transforms import Pad

from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
)

# Constants
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = [3, 3]  # [type_classes, grade_classes]

# Data transforms
def get_transforms(img_size):
    return transforms.Compose([
        transforms.CenterCrop(336),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class BreastCancerDataset(torch.utils.data.Dataset):
    def __init__(self, data_root: pathlib.Path, img_size=384, transform=None):
        self.images_path = list(data_root.glob('**/*.*'))
        self.images_class = [self._get_type_grade(path) for path in self.images_path]
        self.img_size = img_size
        self.transform = transform


    def _get_type_grade(self, path):
        type_grade = path.parent.name

        if type_grade == 'normal':
            return 0, 0
        elif type_grade == 'tis-1':
            return 1, 0
        elif type_grade == 'tis-2':
            return 1, 1
        elif type_grade == 'tis-3':
            return 1, 2
        elif type_grade == 'it-1':
            return 2, 0
        elif type_grade == 'it-2':
            return 2, 1
        elif type_grade == 'it-3':
            return 2, 2

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img = Image.open(self.images_path[idx])
        if img.mode != 'RGB':
            raise ValueError(f"Image {self.images_path[idx]} isn't RGB mode.")
        
        label = self.images_class[idx]
        
        # Handle small images
        w, h = img.size
        min_size = self.img_size
        if w < min_size or h < min_size:
            new_h = min_size if h < min_size else h
            new_w = min_size if w < min_size else w
            img = Pad((new_w//2, new_h//2, new_w - new_w//2, new_h - new_h//2),
              padding_mode='reflect')(img)

        if self.transform:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

class IBOTMultiTaskModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights_path = '/home/yuxin/Downloads/ibot_vit_base_pancan.pth'
        self.base_model = iBOTViT(architecture="vit_base_pancan", encoder="teacher", weights_path=weights_path)
        self.num_features = 768
        self.num_classes = num_classes
        self.heads = nn.ModuleList([nn.Linear(self.num_features, num_class) for num_class in num_classes])

    def forward(self, x, **kwargs):
        x = self.base_model(x)
        return [head(x) for head in self.heads]

class SwinMultiTaskModel(nn.Module):
    def __init__(self, num_classes):
        super(SwinMultiTaskModel, self).__init__()
        self.base_model = timm.create_model('swinv2_base_window12to16_192to256.ms_in22k_ft_in1k', pretrained=False)
        
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
        x = self.base_model.forward_features(x)
        x = self.base_model.feature(x)

        if isinstance(self.num_classes, list):
            x = [head(x) for head in self.heads]
        else:
            x = self.head(x)
        return x
        
def numpy_softmax(logits: np.ndarray) -> np.ndarray:
    # logits: shape (n_samples, n_classes)
    # subtract max per row for numerical stability
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def evaluate_model(model, data_loader, device, tasks):
    """
    Evaluate model on specified tasks.
    Args:
        model: The model to evaluate
        data_loader: DataLoader containing the evaluation data
        device: Device to run evaluation on
        tasks: List of tasks to evaluate. Each task can be:
            - 'type': Full type classification (normal, nonIBC, IBC)
            - 'nonibc': NonIBC grade classification (only for nonIBC cases)
            - 'ibc': IBC grade classification (only for IBC cases)
    """
    model.eval()
    y_true = {task: [] for task in tasks}
    y_pred = {task: [] for task in tasks}
    y_prob = {task: [] for task in tasks}
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, file=sys.stdout):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(x=images)
            
            # Handle type classification
            if 'type' in tasks:
                type_output = outputs[0]
                type_pred = torch.max(type_output, dim=1)[1]
                type_label = labels[:, 0]
                y_pred['type'].extend(type_pred.cpu().tolist())
                y_true['type'].extend(type_label.cpu().tolist())
                y_prob['type'].extend(numpy_softmax(type_output.cpu().numpy()))
            # Handle grade classification for nonIBC
            if 'nonibc' in tasks:
                nonibc_mask = labels[:, 0] == 1  # nonIBC cases
                if nonibc_mask.any():
                    grade_output = outputs[1][nonibc_mask]
                    grade_pred = torch.max(grade_output, dim=1)[1]
                    grade_label = labels[nonibc_mask, 1]
                    y_pred['nonibc'].extend(grade_pred.cpu().tolist())
                    y_true['nonibc'].extend(grade_label.cpu().tolist())
                    y_prob['nonibc'].extend(numpy_softmax(grade_output.cpu().numpy()))
            # Handle grade classification for IBC
            if 'ibc' in tasks:
                ibc_mask = labels[:, 0] == 2  # IBC cases
                if ibc_mask.any():
                    grade_output = outputs[1][ibc_mask]
                    grade_pred = torch.max(grade_output, dim=1)[1]
                    grade_label = labels[ibc_mask, 1]
                    y_pred['ibc'].extend(grade_pred.cpu().tolist())
                    y_true['ibc'].extend(grade_label.cpu().tolist())
                    y_prob['ibc'].extend(numpy_softmax(grade_output.cpu().numpy()))
    
    metrics = {}
    for task in tasks:
        if len(y_true[task]) > 0:  # Only calculate metrics if we have predictions
            metrics[task] = {
                'balanced_acc': round(balanced_accuracy_score(y_true[task], y_pred[task]), 4),
                'kappa': round(cohen_kappa_score(y_true[task], y_pred[task], weights='linear'), 4),
                # 'macro_auroc': round(roc_auc_score(y_true[task], y_prob[task], average='macro', multi_class='ovr'), 4),
                'avg_auc': round(compute_mean_one_vs_all_auc(np.array(y_true[task]), np.array(y_prob[task])), 4),
                'f1_macro': round(f1_score(y_true[task], y_pred[task], average='macro'), 4)
            }
        else:
            metrics[task] = {
                'balanced_acc': None,
                'kappa': None,
                'f1_macro': None
            }
    
    return metrics

def load_model(model_name, weight_path):
    if model_name == 'ibot':
        model = IBOTMultiTaskModel(NUM_CLASSES).to(DEVICE)
        img_size = 384
    elif model_name == 'swin':
        model = SwinMultiTaskModel(NUM_CLASSES)
        model.cuda(0)
        model = nn.DataParallel(model)
        model.cuda(0)
        img_size = 256
    elif model_name =='lora':
        base_model = IBOTMultiTaskModel(NUM_CLASSES)
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=True,
            r=8,
            lora_alpha=16,
            lora_dropout=0.2,
            target_modules=["qkv"],
        )
        model = get_peft_model(base_model, lora_config)
        device = "cuda:0"
        model.to(device)
        img_size = 384
    else:
        raise ValueError(f"Model {model_name} not found")

    model = nn.DataParallel(model,device_ids=[0])
    model.cuda(0)

    ckpt = torch.load(weight_path, map_location=DEVICE)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.cuda(0)
    return model, img_size


def test(model_name, weight_path, data_root, cohort_tasks):
    """
    Main evaluation function.
    Args:
        weight_path: Path to the model weights file.
    """
    # Initialize model
    weight_path = weight_path if isinstance(weight_path, Path) else Path(weight_path)

    model, img_size = load_model(model_name, weight_path)
    data_transforms = get_transforms(img_size)
    
    log_file = weight_path.parent / f'evaluation_{time.strftime("%Y_%m_%d_%H_%M_%S")}.txt'

    # save this script
    script_path = pathlib.Path(__file__)
    shutil.copy(script_path, log_file.parent)
    
    # Evaluate on each cohort
    with open(log_file, 'w') as f:
        for cohort, tasks in cohort_tasks.items():
            print(f"\nEvaluating on {cohort}...")
            dataset = BreastCancerDataset(data_root / cohort / 'test', img_size, data_transforms)
            dataloader = DataLoader(
                dataset,
                batch_size=16,
                shuffle=False,
                pin_memory=True,
                num_workers=0,
                collate_fn=dataset.collate_fn
            )
            
            metrics = evaluate_model(model, dataloader, DEVICE, tasks)
            
            # write and print results
            f.write(f"\n{cohort.upper()}:")
            print(f"\n{cohort.upper()}:", end="")
            for task, task_metrics in metrics.items():
                f.write(f"\n{task.upper()}:")
                print(f"\n{task.upper()}:", end="")
                for metric_name, value in task_metrics.items():
                    if value is not None:
                        f.write(f"\t{metric_name}: {value:.4f}")
                        print(f"\t{metric_name}: {value:.4f}", end="")
                    else:
                        print(f"\t{metric_name}: N/A", end="")

if __name__ == "__main__":
    # data root
    data_root = Path('/mnt/hd0/project/bcacad/data/patch-level')

    # Define tasks for each cohort
    cohort_tasks = {
        'suqh': ['type', 'nonibc', 'ibc'],
        'qduh': ['type', 'nonibc', 'ibc'],
        'shsu': ['type', 'nonibc', 'ibc'],
        'bracs': ['type'],
        'bcnb': ['ibc'],
        'bach': ['type'],
        'apght': ['ibc'],
        'aggregate': ['type', 'nonibc', 'ibc']
    }


    test_model_paths = {
        # 'full_phikon_multi_task_ft': ['ibot', "/mnt/hd1/bcacad/frozen_pretrain_ft/2025_05_06_11_44_15/model-24.pth"],
         'ft_lora': ['lora', "/mnt/hd1/bcacad/timm_lora_ft/2025_07_14_04_18_54/model-13.pth"],
        # 'phikon_ft_multi_task_ft': ['ibot', "/mnt/hd0/project/bcacad/model/pretrainSSL_ibot_vit+ibot_ft+fsl_ft/model-5.pth"],
        # 'lora': ['lora', '/mnt/hd1/bcacad/timm_lora/2025_07_10_21_19_13/model-16.pth'],
        # 'phikon_ft_multi_task_ft': ['ibot', "/mnt/hd1/bcacad/frozen_ftssl_ft/2025_05_11_23_51_43/model-18.pth"],
         #'old_ibot_ft': ['ibot', "/mnt/hd0/project_large_files/bca_grading/suqh/ibot_multi_task/2023_12_28_15_31_01/model-7.pth"],
        # 'true_freeze_ft': ['ibot', "/mnt/hd1/bcacad/frozen_pretrain_ft_truely_checked/2025_07_13_10_36_30/model-15.pth"],
        # 'swinv2_multi_task_ft': ['swin', "/mnt/hd0/project_large_files/bca_grading/suqh/swin_multi_task/2024_01_03_09_24_37/model-14.pth"],
    }

    # Run evaluation for each model
    for test_name, (model_name, model_path) in test_model_paths.items():
        print(f"\nEvaluating {test_name}, {model_name}...")
        test(model_name, model_path, data_root, cohort_tasks)
