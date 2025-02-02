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
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, cohen_kappa_score, accuracy_score
import torch.nn.functional as F
import sys
import time

import shutil
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import time
from sklearn.preprocessing import label_binarize

# from model import swin_tiny_patch4_window7_224 as create_model
from model import swin_base_patch4_window7_224 as create_model

def copy_self(destination):
    # 获取当前脚本的绝对路径
    script_path = os.path.realpath(__file__)
    
    # 拷贝文件到目标位置
    shutil.copy2(script_path, destination)


device = "cuda:0" if torch.cuda.is_available() else "cpu"

img_size = patch_size = 224
data_root = pathlib.Path('/home/yuxin/ssd_data/SUQH_10x336')
phases = ['train', 'val', 'test']
save_root = pathlib.Path('/mnt/hd0/project_large_files/bca_grading/suqh/swin_multi_task')
save_root.mkdir(exist_ok=True)
data_trans = {
    "train": transforms.Compose([
                                #  transforms.RandomResizedCrop(img_size),
                                # transforms.RandomCrop(img_size*4, padding_mode='reflect', pad_if_needed=True),
                                # transforms.CenterCrop(img_size),
                                #  transforms.CenterCrop(img_size*4),
                                # transforms.Resize(224),
                                # RandomPatchCrop(img_size*4),
                                transforms.Resize(img_size),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "test": transforms.Compose([
                                # transforms.Resize(int(img_size * 1.143)),
                                # transforms.CenterCrop(img_size*4),
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
# ibot_base_pancancer.head = torch.nn.Linear(768, num_classes).to(device)
# ibot_base_pancancer.head.weight.data.normal_(mean=0.0, std=0.02)
# ibot_base_pancancer.head.bias.data.zero_()

model = create_model(num_classes=num_classes).to(device)

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


bs=64
# 实例化训练数据集
train_dataset = MyDataSet(data_root=data_root,
                            phase='train',
                            transform=data_trans["train"])

# 实例化验证数据集
val_dataset = MyDataSet(data_root=data_root,
                        phase='val',
                        transform=data_trans["test"])

batch_size = bs
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])  # number of workers
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
optimizer = optim.AdamW(pg, lr=1e-3, weight_decay=5E-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-6)
# optimizer = optim.AdamW(pg, lr=1e-4, weight_decay=5E-2)
best_loss = np.inf
best_f1 = 0
epochs_without_improvement = 0
patience = 3
epochs = 100


# In[18]:


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


        data_loader.desc = "[{} epoch {}] loss: {:.3f}, acc: {:.3f},{:.3f}, f1_marco: {:.3f},{:.3f}".format(
            phase, epoch, accu_loss.item() / (step + 1), acc[0], acc[1], f1_macro[0], f1_macro[1])

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        if phase == 'train':
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

    return accu_loss.item() / (step + 1), acc, f1_macro, y_true, y_pred, y_prob


log_time = time.strftime("%Y_%m_%d_%H:%M:%S")
log_filename = 'train_'+ log_time + '.txt'
log_filepath = save_root / log_time / log_filename
weight_save_path = save_root / log_time
tb_writer = SummaryWriter()

for epoch in range(epochs):
    # train
    train_loss, train_acc, train_f1_macro, _, _, _ = iterate(model=model,
                                                                optimizer=optimizer,
                                                                data_loader=train_loader,
                                                                device=device,
                                                                phase='train', epoch=epoch)

    # validate
    val_loss, val_acc, val_f1_macro,_,_,_= iterate(model=model,
                                                    data_loader=val_loader,
                                                    device=device,
                                                    epoch=epoch,phase='val')

    # tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
    # tb_writer.add_scalar(tags[0], train_loss, epoch)
    # tb_writer.add_scalar(tags[1], train_acc, epoch)
    # tb_writer.add_scalar(tags[2], val_loss, epoch)
    # tb_writer.add_scalar(tags[3], val_acc, epoch)
    # tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

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
    save_root.mkdir(exist_ok=True)
    weight_save_path.mkdir(exist_ok=True)

    with open(log_filepath, "a") as f:
        f.write(to_write)

    improve = False
    if val_loss < best_loss:
        best_loss = val_loss
        epochs_without_improvement = 0
        improve = True
        torch.save(model.state_dict(), os.path.join(weight_save_path, 'model-{}.pth').format(epoch))

    # if val_f1_macro > best_f1:
    #     best_f1 = val_f1_macro
    #     epochs_without_improvement = 0
    #     improve = True
    #     torch.save(model.state_dict(), os.path.join(weight_save_path, 'model-{}.pth').format(epoch))
    
    if improve is False:
        epochs_without_improvement += 1

    if epochs_without_improvement == patience:
        print('Early stopping at epoch {}...'.format(epoch+1))
        torch.save(model.state_dict(), os.path.join(weight_save_path, 'model-{}.pth').format(epoch))
        break

weights_dir = weight_save_path
logs_save_path = weights_dir

log_filepath.parent.mkdir(exist_ok=True)
copy_self(save_root/log_time/'run.py')

if os.path.exists(logs_save_path) is False:
    os.makedirs(logs_save_path)
log_filename = 'test_'+time.strftime("%Y_%m_%d_%H:%M:%S") + '.txt'
log_filepath = os.path.join(logs_save_path, log_filename)

for weights_path in weights_dir.rglob("*.pth"):
    # print(weights_path)
    # continue

    model_weight_path = weights_path
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    # validate
    _,_,_,y_true,y_pred,y_prob  = iterate(model=model,
                data_loader=test_loader,
                device=device,
                epoch=weights_path.name)

    # class_level
    Acc = {}
    F1score = {}
    Auc = {}
    Kappa = {}
    Precision = {}
    Recall = {}
    for task in ['type', 'grade']:
        Acc[task] = accuracy_score(y_true[task], y_pred[task])
        y_true_b = label_binarize(y_true[task], classes=[0, 1, 2])
        Auc[task] = roc_auc_score(y_true_b, y_prob[task], multi_class='ovo', average=None)
        Kappa[task] = cohen_kappa_score(y_true[task], y_pred[task], weights='quadratic')
        Precision[task], Recall[task], F1score[task], _ = precision_recall_fscore_support(y_true[task], y_pred[task])


    
    log_txt_formatter = "{weights_n}: \n" \
        "[Type] [AUC] {auc_t} [ACC] {acc_t} [F1-Score] {f1_t} [Kappa] {kappt_t} [Precision] {p_t} [Recall] {r_t}\n" \
        "[Grade] [AUC] {auc_g} [ACC] {acc_g} [F1-Score] {f1_g} [Kappa] {kappt_g} [Precision] {p_g} [Recall] {r_g}\n\n"

    to_write = log_txt_formatter.format(weights_n="{: <12}".format(weights_path.name),
                                        acc_t = Acc['type'],
                                        f1_t = F1score['type'],
                                        auc_t = Auc['type'],
                                        kappt_t = Kappa['type'],
                                        p_t = Precision['type'],
                                        r_t = Recall['type'],
                                        acc_g = Acc['grade'],
                                        f1_g = F1score['grade'],
                                        auc_g = Auc['grade'],
                                        kappt_g = Kappa['grade'],
                                        p_g = Precision['grade'],
                                        r_g = Recall['grade'],    
                                    )
    print(to_write)
    with open(log_filepath, "a") as f:
        f.write(to_write)

