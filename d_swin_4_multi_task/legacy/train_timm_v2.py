#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# from model import swin_tiny_patch4_window7_224 as create_model
# from model import swin_base_patch4_window7_224 as create_model

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def copy_self(destination):
    # 获取当前脚本的绝对路径
    script_path = os.path.realpath(__file__)
    
    # 拷贝文件到目标位置
    shutil.copy2(script_path, destination)


device = "cuda:0" if torch.cuda.is_available() else "cpu"

img_size = patch_size = 256
# data_root = pathlib.Path('/home/dc/yuxinwu/breast_cad/SUQH_10x336')
data_root = pathlib.Path('/home/yuxin/ssd_data/SUQH_10x336')
phases = ['train', 'val', 'test']
# save_root = pathlib.Path('/media/dc/sdb1/yuxinwu/swin_multi_task')
save_root = pathlib.Path('/mnt/hd0/project_large_files/bca_grading/suqh/swin_multi_task')
save_root.mkdir(exist_ok=True)
data_trans = {
    "train": transforms.Compose([
                                transforms.RandomResizedCrop(size=img_size, scale=(0.8, 1.2)),
                                transforms.Resize(img_size),
                                transforms.RandomRotation(degrees=15),
                                transforms.ColorJitter(),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "test": transforms.Compose([
                                transforms.Resize(img_size),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


# In[4]:


class MyDataSet(Dataset):
    """自定义数据集"""

    # def __init__(self, images_path: list, images_class: list, images_ncl: list, images_epi: list, images_tub: list, images_mit: list, transform=None):
    
    def get_type_grade(self, path):
        type_grade = path.parent.name
        if type_grade == 'normal':
            return 0,0
        elif type_grade == 'tis-1':
            return 1,0
        elif type_grade == 'tis-2':
            return 1,1
        elif type_grade == 'tis-3':
            return 1,2
        elif type_grade == 'it-1':
            return 2,0
        elif type_grade == 'it-2':
            return 2,1
        elif type_grade == 'it-3':
            return 2,2
    
    def __init__(self, data_root:pathlib, phase:str, transform=None):
        self.images_path = list((data_root/phase).glob('**/*.jpg'))
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


# In[5]:


num_classes = [3,3]

class TransMultiTaskModel(nn.Module):
    def __init__(self, base_model_name, num_classes):
        super(MultiTaskModel, self).__init__()
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

class ConvMultiTaskModel(nn.Module):
    def __init__(self, base_model_name, num_classes):
        super(MultiTaskModel, self).__init__()
        ckpt_dict = {
            'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k': dict(file='/home/dc/yuxinwu/breast_cad/swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth'),
            'convnext_large.fb_in22k_ft_in1k_384': dict(file='/home/dc/yuxinwu/breast_cad/convnext_large_22k_1k_384.pth'),
            'convnext_base.fb_in22k_ft_in1k_384': dict(file='/home/dc/yuxinwu/breast_cad/convnext_base_22k_1k_384.pth'),
        }
        self.base_model = timm.create_model(base_model_name, pretrained=True, pretrained_cfg_overlay=ckpt_dict[base_model_name])
        # print(self.base_model)
        self.num_features = self.base_model.num_features
        self.num_classes = num_classes

        self.base_model.feature = nn.Sequential(
            self.base_model.head.global_pool,
            self.base_model.head.norm,
            self.base_model.head.flatten,
            self.base_model.head.pre_logits,
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



# model_name = 'swin_base_patch4_window12_384_in22k'
# model_name = 'swin_base_patch4_window7_224_in22k'
model_name = 'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k'
# model_name = 'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k'
# model_name = 'swinv2_large_window12to24_192to384.ms_in22k_ft_in1k'
# model_name = 'convnext_large.fb_in22k_ft_in1k_384'
# model_name = 'convnext_base.fb_in22k_ft_in1k_384'
# MultiTaskModel = ConvMultiTaskModel
MultiTaskModel = TransMultiTaskModel

model = MultiTaskModel(model_name, num_classes)

device = "cuda"
model.to(device)

# model = nn.DataParallel(model)
# model.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state_dict'])
# ckpt = torch.load(ckpt_path, map_location=device)
# model.load_state_dict(ckpt)
# for name, param in model.base_model.state_dict().items():
#     if f'module.base_model.{name}' in ckpt:
#         if param.size() == ckpt[f'module.base_model.{name}'].size():
#             print('load {}'.format(name))
#             model.base_model.state_dict()[name].copy_(ckpt[f'module.base_model.{name}'])
#         else:
#             print('size mismatch: {}, {}, {}'.format(name, param.size(), ckpt[f'module.base_model.{name}'].size()))

# for name, para in model.named_parameters():
#     para.requires_grad_(True)
#     print("training {}".format(name))

for name, para in model.named_parameters():
    para.requires_grad_(True)
    # if "head" not in name and 'blocks.11' not in name:
    #     para.requires_grad_(False)
    # else:
    #     para.requires_grad_(True)
    #     print("training {}".format(name))


# In[6]:


# 实例化训练数据集
train_dataset = MyDataSet(data_root=data_root,
                            phase='train',
                            transform=data_trans["train"])

# 实例化验证数据集
val_dataset = MyDataSet(data_root=data_root,
                        phase='val',
                        transform=data_trans["test"])

bs=16
batch_size = bs
nw = 4
print('Using {} dataloader workers every process'.format(nw))
train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=nw,
                                            collate_fn=train_dataset.collate_fn)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=batch_size//2,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=nw,
                                            collate_fn=val_dataset.collate_fn)

# test set
test_dataset = MyDataSet(data_root=data_root, phase='test',
                            transform=data_trans['test'])

test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size//2,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=nw,
                                            collate_fn=test_dataset.collate_fn)
# In[7]:


# model


# In[8]:
pg = [p for p in model.parameters() if p.requires_grad]
base_optimizer =torch.optim.AdamW # define an optimizer for the "sharpness-aware" update
# lr = 5e-6
lr = 5e-5
optimizer = SAM(pg, base_optimizer,lr=lr, betas=(0.9, 0.999), weight_decay=0.05)
loss_function = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)
# model = nn.DataParallel(model)

best_loss = np.inf
best_f1 = 0
epochs_without_improvement = 0
patience = 3
epochs = 20

model = nn.DataParallel(model)
model = nn.DataParallel(model)

# ckpt_path = '/mnt/hd0/project_large_files/bca_grading/suqh/swin_multi_task/2023_12_15_18:17:54/model-34.pth'
ckpt_path = '/mnt/hd0/project_large_files/bca_grading/suqh/swin_multi_task/2024_01_02_17_39_16/model-9.pth'
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])

# In[18]:


# def iterate(model, data_loader, device, optimizer=None, loss_function=None, phase='test', epoch='Test'):
#     if phase == 'train':
#         model.train()
#         optimizer.zero_grad()
#     else:
#         model.eval()
    
#     # loss_function = torch.nn.CrossEntropyLoss()
#     accu_loss = torch.zeros(1).to(device)  # 累计损失
#     sample_num = 0
#     tasks = ['type', 'grade']
#     y_true, y_pred, y_prob = {}, {}, {}
#     for t in tasks:
#         y_true[t], y_pred[t], y_prob[t] = [], [], []
#     data_loader = tqdm(data_loader, file=sys.stdout)
#     for step, data in enumerate(data_loader):
#         images, labels = data
#         images = images.to(device)
#         labels = labels.to(device)
#         sample_num += images.shape[0]

#         outputs = model.forward(images)

#         losses = []

#         for i in range(len(outputs)):
#             task = tasks[i]
#             if task == 'grade':
#                 output = outputs[i]
#                 pred = torch.max(output, dim=1)[1]
#                 label = labels[:,i]
#                 mask = labels[:,0] != 0
#                 loss = loss_function(output, label) * mask.float()
#                 loss = torch.mean(loss)
#                 pred = pred[mask]
#                 label = label[mask]
#                 output = output[mask]
#             else: 
#                 output = outputs[i]
#                 pred = torch.max(output, dim=1)[1]
#                 label = labels[:,i]
#                 loss = loss_function(output, label)
#                 loss = torch.mean(loss)
#             losses.append(loss)


#             y_pred[task].extend(pred.tolist())
#             y_true[task].extend(label.to(device).tolist())
            
#             prob = F.softmax(output, dim=1)
#             y_prob[task].extend(prob.to(device).tolist())
        
#         loss = sum(losses) / len(losses)
        
#         accu_loss += loss.detach()

#         # compute acc
#         acc = []
#         f1_macro = []
#         for task in tasks:
#             acc.append(accuracy_score(y_true[task], y_pred[task]))
#             f1_macro.append(f1_score(y_true[task], y_pred[task], average='macro'))


#         data_loader.desc = "[{} epoch {}] loss: {:.3f}, acc: {:.3f},{:.3f}, f1_marco: {:.3f},{:.3f}".format(
#             phase, epoch, accu_loss.item() / (step + 1), acc[0], acc[1], f1_macro[0], f1_macro[1])

#         if not torch.isfinite(loss):
#             print('WARNING: non-finite loss, ending training ', loss)
#             sys.exit(1)
#         if phase == 'train':
#             loss.backward()
#             optimizer.first_step(zero_grad=True)
#             outputs = model.forward(images)
#             losses = []
#             for i in range(len(outputs)):
#                 task = tasks[i]
#                 if task == 'grade':
#                     output = outputs[i]
#                     label = labels[:,i]
#                     mask = labels[:,0] != 0
#                     loss = loss_function(output, label) * mask.float()
#                     loss = torch.mean(loss)
#                 else: 
#                     output = outputs[i]
#                     label = labels[:,i]
#                     loss = loss_function(output, label)
#                     loss = torch.mean(loss)
#                 losses.append(loss)

#             loss = sum(losses) / len(losses)
#             loss.backward()
#             optimizer.second_step(zero_grad=True)
#             scheduler.step()

#     return accu_loss.item() / (step + 1), acc, f1_macro, y_true, y_pred, y_prob

def iterate(model, data_loader, device, optimizer=None, loss_function=None, phase='test', epoch='Test'):
    if phase == 'train':
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()
    
    # loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    sample_num = 0
    tasks = ['type', 'grade']
    y_true, y_pred, y_prob = {}, {}, {}
    for t in tasks:
        y_true[t], y_pred[t], y_prob[t] = [], [], []
    data_loader = tqdm(data_loader, file=sys.stdout)
    a = 0.5
    for step, data in enumerate(data_loader):
        images, labels = data
        # print(labels)
        images = images.to(device)
        labels = labels.to(device)
        sample_num += images.shape[0]

        outputs = model.forward(images)

        losses = []

        for i in range(len(outputs)):
            task = tasks[i]
            if task == 'grade':
                output = outputs[i]
                pred = torch.max(output, dim=1)[1]
                label = labels[:,i]
                mask = labels[:,0] != 0
                loss = loss_function(output, label) * mask.float()
                loss = torch.mean(loss)
                # pred = pred[mask]
                # label = label[mask]
                # output = output[mask]
            else: 
                output = outputs[i]
                pred = torch.max(output, dim=1)[1]
                label = labels[:,i]
                loss = loss_function(output, label)
                loss = torch.mean(loss)
            losses.append(loss)


            y_pred[task].extend(pred.tolist())
            y_true[task].extend(label.to(device).tolist())
            
            prob = F.softmax(output, dim=1)
            y_prob[task].extend(prob.to(device).tolist())
        
        loss = (1-a)*losses[0] + a*losses[1]
        
        accu_loss += loss.detach()

        # compute acc
        acc = []
        f1_macro = []
        for task in tasks:
            if task != 'type':
                tumor_idx = np.where(np.array(y_true['type']) > 0)[0]
                y_true_ = np.array(y_true[task])[tumor_idx]
                y_pred_ = np.array(y_pred[task])[tumor_idx]
                acc.append(accuracy_score(y_true_, y_pred_))
                f1_macro.append(f1_score(y_true_, y_pred_, average='macro'))
            else:    
                acc.append(accuracy_score(y_true[task], y_pred[task]))
                f1_macro.append(f1_score(y_true[task], y_pred[task], average='macro'))

        a = f1_macro[0]/(f1_macro[0]+f1_macro[1])
        data_loader.desc = "[{} epoch {}] loss: {:.3f}, acc: {:.3f},{:.3f}, f1_marco: {:.3f},{:.3f}".format(
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
                task = tasks[i]
                if task == 'grade':
                    output = outputs[i]
                    label = labels[:,i]
                    mask = labels[:,0] != 0
                    loss = loss_function(output, label) * mask.float()
                    loss = torch.mean(loss)
                else: 
                    output = outputs[i]
                    label = labels[:,i]
                    loss = loss_function(output, label)
                    loss = torch.mean(loss)
                losses.append(loss)

            loss = (1-a)*losses[0] + a*losses[1]
            loss.backward()
            optimizer.second_step(zero_grad=True)
            scheduler.step()
    
        # if phase != 'test':
        # if step > 10:
        #     break
    return accu_loss.item() / (step + 1), acc, f1_macro, y_true, y_pred, y_prob

log_time = time.strftime("%Y_%m_%d_%H_%M_%S")
log_filename = 'train_'+ log_time + '.txt'
log_filepath = save_root / log_time / log_filename
weight_save_path = save_root / log_time
weight_save_path.mkdir(exist_ok=True)
tb_writer = SummaryWriter()
copy_self(save_root/log_time/'run.py')

for epoch in range(epochs):
    # train
    train_loss, train_acc, train_f1_macro, _, _, _ = iterate(model=model,
                                                                optimizer=optimizer,
                                                                loss_function=loss_function,
                                                                data_loader=train_loader,
                                                                device=device,
                                                                phase='train', epoch=epoch)

    # validate
    val_loss, val_acc, val_f1_macro,_,_,_= iterate(model=model,
                                                   loss_function=loss_function,
                                                    data_loader=val_loader,
                                                    device=device,
                                                    epoch=epoch,phase='val')
    
    # write to txt
    log_txt_formatter = "[train Epoch] {epoch:03d} [Loss] {train_loss} [acc] {train_acc} [f1_macro] {train_f1}\n" \
                        "[valid Epoch] {epoch:03d} [Loss] {valid_loss} [acc] {valid_acc} [f1_macro] {valid_f1}\n"

    to_write = log_txt_formatter.format(epoch=epoch,
                                        train_loss=train_loss,
                                        train_acc=train_acc,
                                        train_f1 = train_f1_macro,
                                        valid_loss=val_loss,
                                        valid_acc=val_acc,
                                        valid_f1 = val_f1_macro)
    

    _,_,_,y_true,y_pred,y_prob  = iterate(model=model,
                data_loader=test_loader,
                loss_function=loss_function,
                device=device,
                epoch='test')
    
    Acc = {}
    F1score = {}
    Auc = {}
    Kappa = {}
    Precision = {}
    Recall = {}
    
    # for type
    type_y_pred = y_pred['type']
    type_y_prob = y_prob['type']
    type_y_true = y_true['type']
    Acc['type'] = accuracy_score(type_y_true, type_y_pred)
    type_y_true_b = label_binarize(type_y_true, classes=[0, 1, 2])
    Auc['type'] = roc_auc_score(type_y_true_b, type_y_prob, multi_class='ovo', average=None)
    Kappa['type'] = cohen_kappa_score(type_y_true, type_y_pred, weights='quadratic')
    Precision['type'], Recall['type'], F1score['type'], _ = precision_recall_fscore_support(type_y_true, type_y_pred)

    log_txt_formatter = "[test Epoch] {epoch:03} [Type] [AUC] {auc_t} [ACC] {acc_t} [F1-Score] {f1_t} [Kappa] {kappt_t} [Precision] {p_t} [Recall] {r_t}\n\n"
    to_write = log_txt_formatter.format(epoch=epoch,
                                            acc_t = Acc['type'],
                                            f1_t = F1score['type'],
                                            auc_t = Auc['type'],
                                            kappt_t = Kappa['type'],
                                            p_t = Precision['type'],
                                            r_t = Recall['type'],
                                        )
    print(to_write)
    with open(log_filepath, "a") as f:
        f.write(to_write)
    
    # for grade noIBC
    tumor_idx = np.where(np.array(y_true['type']) == 1)[0]
    grade_y_pred = np.array(y_pred['grade'])[tumor_idx]
    grade_y_prob = np.array(y_prob['grade'])[tumor_idx]
    grade_y_true = np.array(y_true['grade'])[tumor_idx]
    Acc['nonibc'] = accuracy_score(grade_y_true, grade_y_pred)
    grade_y_true_b = label_binarize(grade_y_true, classes=[0, 1, 2])
    Auc['nonibc'] = roc_auc_score(grade_y_true_b, grade_y_prob, multi_class='ovo', average=None)
    Kappa['nonibc'] = cohen_kappa_score(grade_y_true, grade_y_pred, weights='quadratic')
    Precision['nonibc'], Recall['nonibc'], F1score['nonibc'], _ = precision_recall_fscore_support(grade_y_true, grade_y_pred)

    log_txt_formatter = "[test Epoch] {epoch:03d} [Grade-nonIBC] [AUC] {auc_g} [ACC] {acc_g} [F1-Score] {f1_g} [Kappa] {kappt_g} [Precision] {p_g} [Recall] {r_g}\n\n"
    to_write = log_txt_formatter.format(epoch=epoch,
                                        acc_g = Acc['nonibc'],
                                        f1_g = F1score['nonibc'],
                                        auc_g = Auc['nonibc'],
                                        kappt_g = Kappa['nonibc'],
                                        p_g = Precision['nonibc'],
                                        r_g = Recall['nonibc'],    
                                    )
    print(to_write)
    with open(log_filepath, "a") as f:
        f.write(to_write)

    # for grade IBC
    tumor_idx = np.where(np.array(y_true['type']) == 2)[0]
    grade_y_pred = np.array(y_pred['grade'])[tumor_idx]
    grade_y_prob = np.array(y_prob['grade'])[tumor_idx]
    grade_y_true = np.array(y_true['grade'])[tumor_idx]
    Acc['ibc'] = accuracy_score(grade_y_true, grade_y_pred)
    grade_y_true_b = label_binarize(grade_y_true, classes=[0, 1, 2])
    Auc['ibc'] = roc_auc_score(grade_y_true_b, grade_y_prob, multi_class='ovo', average=None)
    Kappa['ibc'] = cohen_kappa_score(grade_y_true, grade_y_pred, weights='quadratic')
    Precision['ibc'], Recall['ibc'], F1score['ibc'], _ = precision_recall_fscore_support(grade_y_true, grade_y_pred)

    log_txt_formatter = "[test Epoch] {epoch:03d} [Grade-IBC] [AUC] {auc_g} [ACC] {acc_g} [F1-Score] {f1_g} [Kappa] {kappt_g} [Precision] {p_g} [Recall] {r_g}\n\n"
    to_write = log_txt_formatter.format(epoch=epoch,
                                        acc_g = Acc['ibc'],
                                        f1_g = F1score['ibc'],
                                        auc_g = Auc['ibc'],
                                        kappt_g = Kappa['ibc'],
                                        p_g = Precision['ibc'],
                                        r_g = Recall['ibc'],    
                                    )
    print(to_write)
    with open(log_filepath, "a") as f:
        f.write(to_write)




    save_root.mkdir(exist_ok=True)
    weight_save_path.mkdir(exist_ok=True)

    improve = False
    if val_loss < best_loss:
        best_loss = val_loss
        epochs_without_improvement = 0
        improve = True
        checkpoint_dict = {'epoch': epoch, 
                   'model_state_dict': model.state_dict(), 
                   'optim_state_dict': optimizer.state_dict(), 
                   'criterion_state_dict': loss_function.state_dict()}
        torch.save(checkpoint_dict, os.path.join(weight_save_path, 'model-{}.pth').format(epoch))
        # torch.save(model.state_dict(), os.path.join(weight_save_path, 'model-{}.pth').format(epoch))

    # if val_f1_macro > best_f1:
    #     best_f1 = val_f1_macro
    #     epochs_without_improvement = 0
    #     improve = True
    #     torch.save(model.state_dict(), os.path.join(weight_save_path, 'model-{}.pth').format(epoch))
    
    if improve is False:
        epochs_without_improvement += 1

    if epochs_without_improvement == patience:
        print('Early stopping at epoch {}...'.format(epoch+1))
        # torch.save(model.state_dict(), os.path.join(weight_save_path, 'model-{}.pth').format(epoch))
        checkpoint_dict = {'epoch': epoch, 
                   'model_state_dict': model.state_dict(), 
                   'optim_state_dict': optimizer.state_dict(), 
                   'criterion_state_dict': loss_function.state_dict()}
        torch.save(checkpoint_dict, os.path.join(weight_save_path, 'model-{}.pth').format(epoch))
        break
