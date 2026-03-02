#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, cohen_kappa_score
import torch.nn.functional as F
import sys
import time

import shutil
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def copy_self(destination):
    # 获取当前脚本的绝对路径
    script_path = os.path.realpath(__file__)
    
    # 拷贝文件到目标位置
    shutil.copy2(script_path, destination)



# In[2]:


device = "cuda:0" if torch.cuda.is_available() else "cpu"
weights_path = '/home/yuxin/Downloads/ibot_vit_base_pancan.pth'
ibot_base_pancancer = iBOTViT(architecture="vit_base_pancan", encoder="student", weights_path=weights_path).to(device)


# In[3]:

scale = 4
patch_size = 224
data_root = pathlib.Path('/mnt/hd0/original_datasets/JT_Breast/SUQH/40xTiles')
split_root = pathlib.Path('/mnt/hd0/project_large_files/bca_grading/suqh/split_data')
phases = ['train', 'val', 'test']
save_root = pathlib.Path('/mnt/hd0/project_large_files/bca_grading/suqh/histosslscaling_finetune')
save_root.mkdir(exist_ok=True)
data_trans = dict(
    test = A.Compose([
                        A.PadIfNeeded(min_height=patch_size*scale, min_width=patch_size*scale, border_mode=0, value=0, p=1),
                        A.CenterCrop(patch_size*scale, patch_size*scale),
                        A.Resize(patch_size, patch_size),
                        A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                        ToTensorV2(),
                    ]),
    train = A.Compose([
                        A.PadIfNeeded(min_height=patch_size*scale, min_width=patch_size*scale, border_mode=0, value=0, p=1),
                        A.CenterCrop(patch_size*scale, patch_size*scale),
                        A.Resize(patch_size, patch_size),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.ChannelDropout(p=0.5),
                        A.ChannelShuffle(p=0.5),
                        A.CoarseDropout(max_holes=8, max_height=8, max_width=8, min_holes=1, min_height=1, min_width=1, fill_value=0, p=0.5),
                        A.ColorJitter(p=0.5),
                        A.ImageCompression(quality_lower=50, quality_upper=100, p=0.5),
                        A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                        ToTensorV2(),
                    ]),
)


# In[4]:


class MyDataSet(Dataset):
    """自定义数据集"""

    # def __init__(self, images_path: list, images_class: list, images_ncl: list, images_epi: list, images_tub: list, images_mit: list, transform=None):
    def __init__(self, data_root:str, split_file:str,transform=None):
        images_path = []
        images_class = []
        with open(split_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                label = line[-2:-1]
                if label == '0' or label == '2':
                    label = 0
                else:
                    label = 1
                img_name = line[:-3]
                img_path = os.path.join(data_root, img_name)
                images_path.append(img_path)
                images_class.append(int(label))
 
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        # img = Image.open(self.images_path[item])
        image = cv2.imread(self.images_path[item])
        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # RGB为彩色图片，L为灰度图片
        # if img.mode != 'RGB':
        #     raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:

            augmented = self.transform(image=image)
            image = augmented['image']



        return image, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)

        return images, labels


# In[5]:


num_classes = 2
ibot_base_pancancer.head = torch.nn.Linear(768, num_classes).to(device)
ibot_base_pancancer.head.weight.data.normal_(mean=0.0, std=0.02)
ibot_base_pancancer.head.bias.data.zero_()


model = ibot_base_pancancer

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


bs=14
# 实例化训练数据集
train_dataset = MyDataSet(data_root=data_root,
                            split_file=split_root / 'train.txt',
                            transform=data_trans["train"])

# 实例化验证数据集
val_dataset = MyDataSet(data_root=data_root,
                        split_file=split_root / 'val.txt',
                        transform=data_trans["test"])

batch_size = bs
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
print('Using {} dataloader workers every process'.format(nw))
train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=nw,
                                            collate_fn=train_dataset.collate_fn)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=nw,
                                            collate_fn=val_dataset.collate_fn)


# In[7]:


# model


# In[8]:


pg = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.AdamW(pg, lr=1e-5, weight_decay=5E-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-6)
# optimizer = optim.AdamW(pg, lr=1e-4, weight_decay=5E-2)
best_loss = np.inf
epochs_without_improvement = 0
patience = 20
# patience = 2
epochs = 100


# In[18]:


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    y_true, y_pred, y_score_auc = [], [], []
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        # images, labels, ncl, epi, tub, mit = data
        images, labels = data
        sample_num += images.shape[0]

        pred = model.forward(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        # print(loss, pred.shape, labels)
        loss.backward()
        accu_loss += loss.detach()

        #################################
        # print(pred_classes.tolist())
        y_pred.extend(pred_classes.tolist())
        # print(y_pred)
        # print(type(y_pred))

        # print(labels.to(device).tolist())
        y_true.extend(labels.to(device).tolist())
        # print(y_true)
        # print(type(y_true))
        f1_macro=f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        ##auc
        pred_auc = F.softmax(pred, dim=1)
        y_score_auc.extend(pred_auc.to(device).tolist())
        ################end################

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, f1_a: {:.3f}, f1_i: {:.3f}".format(epoch,
                                                                                                           accu_loss.item() / (step + 1),
                                                                                                           accu_num.item() / sample_num,
                                                                                                           f1_macro,
                                                                                                           f1_micro)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        # break

    # y_true2 = label_binarize(y_true, classes=[0, 1, 2])
    # auc_class = roc_auc_score(y_true2, y_score_auc, multi_class="ovo", average=None)
    # auc1 = roc_auc_score(y_true2, y_score_auc, multi_class="ovo", average="macro")
    
    y_score_auc = np.array(y_score_auc)
    y_true = np.array(y_true)
    # print(y_true)
    # print(y_score_auc)
    # print(y_true.shape, y_score_auc.shape)

    

    auc_score = roc_auc_score(y_true, y_score_auc[:,1])
    auc1 = auc_class = auc_score

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, auc1, f1_macro, f1_micro, auc_class


# In[22]:


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    y_true, y_pred, y_score_auc = [], [], []
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model.forward(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()



        loss = loss_function(pred, labels.to(device))
        # print(loss)
        accu_loss += loss

        ################################
        # print(pred_classes.tolist())
        y_pred.extend(pred_classes.tolist())
        # print(y_pred)
        # print(type(y_pred))
        y_true.extend(labels.to(device).tolist())
        # print(y_true)
        # print(type(y_true))
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        ##auc
        pred_auc = F.softmax(pred, dim=1)
        y_score_auc.extend(pred_auc.to(device).tolist())
        ################end################

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}, f1_a: {:.3f}, f1_i: {:.3f}".format(epoch,
                                                                                                           accu_loss.item() / (
                                                                                                                       step + 1),
                                                                                                           accu_num.item() / sample_num,
                                                                                                           f1_macro,
                                                                                                           f1_micro)
        # break

    # y_true2 = label_binarize(y_true, classes=[0, 1, 2])
    # auc_class = roc_auc_score(y_true2, y_score_auc, multi_class="ovo", average=None)
    # auc1 = roc_auc_score(y_true2, y_score_auc, multi_class="ovo", average="macro")
    y_score_auc = np.array(y_score_auc)
    y_true = np.array(y_true)
    auc_score = roc_auc_score(y_true, y_score_auc[:,1])
    auc1 = auc_class = auc_score

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, auc1, f1_macro, f1_micro, auc_class

@torch.no_grad()
def test(model, data_loader, device, weight_name):

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    y_true, y_pred, y_score_auc = [], [], []
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        # print('--------------',step+1,'-----------------')
        # images, labels, ncl, epi, tub, mit = data
        images, labels = data
        # print(labels)
        sample_num += images.shape[0]

        pred = model.forward(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        #################################
        y_pred.extend(pred_classes.tolist())
        y_true.extend(labels.to(device).tolist())
        f1_macro=f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        qwk = cohen_kappa_score(y_true, y_pred, labels=None, weights='quadratic', sample_weight=None)

        pred_auc = F.softmax(pred, dim=1)
        y_score_auc.extend(pred_auc.to(device).tolist())
        ################end################

        data_loader.desc = "weight_name: {}, acc: {:.4f}, f1_a: {:.4f}, q-w-k: {:.4f}".format(weight_name,
                                                                                              accu_num.item() / sample_num,
                                                                                              f1_macro,
                                                                                              qwk, )

    p_class, r_class, f_class, support_micro = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred,
                                                                               average=None)

    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    y_score_auc = np.array(y_score_auc)
    y_true = np.array(y_true)
    # y_true2 = label_binarize(y_true, classes=[0, 1, 2])
    # auc_class = roc_auc_score(y_true2, y_score_auc, multi_class="ovo", average=None)
    # auc1 = roc_auc_score(y_true2, y_score_auc, multi_class="ovo", average="macro")
    auc_score = roc_auc_score(y_true, y_score_auc[:,1])
    auc1 = auc_class = auc_score

    # test_loss, test_auc, test_acc, f1_macro, qwk, auc_class, p_class, r_class, f_class
    return accu_loss.item() / (step + 1), auc1, f1_micro, f1_macro, qwk, auc_class, p_class, r_class, f_class
# In[24]:


log_time = time.strftime("%Y_%m_%d_%H:%M:%S")
log_filename = 'train_'+ log_time + '.txt'
log_filepath = save_root / log_time / log_filename
weight_save_path = save_root / log_time




tb_writer = SummaryWriter()
for epoch in range(epochs):
    # train
    train_loss, train_acc, train_auc, train_f1_macro, train_f1_micro, train_auc_class = train_one_epoch(model=model,
                                                                                                        optimizer=optimizer,
                                                                                                        data_loader=train_loader,
                                                                                                        device=device,
                                                                                                        epoch=epoch,)

    # validate
    val_loss, val_acc, val_auc, val_f1_macro,val_f1_micro, val_auc_class= evaluate(model=model,
                                                                                    data_loader=val_loader,
                                                                                    device=device,
                                                                                    epoch=epoch)

    tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
    tb_writer.add_scalar(tags[0], train_loss, epoch)
    tb_writer.add_scalar(tags[1], train_acc, epoch)
    tb_writer.add_scalar(tags[2], val_loss, epoch)
    tb_writer.add_scalar(tags[3], val_acc, epoch)
    tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

    # write to txt
    log_txt_formatter = "[train Epoch] {epoch:03d} [Loss] {train_loss} [auc] {train_auc} [acc] {train_acc} [f1_macro] {train_f1_ma} [f1_micro] {train_f1_mi} [AUC] {train_auc_c}\n" \
                        "[valid Epoch] {epoch:03d} [Loss] {valid_loss} [auc] {valid_auc} [acc] {valid_acc} [f1_macro] {valid_f1_ma} [f1_micro] {valid_f1_mi} [AUC] {valid_auc_c}\n"

    to_write = log_txt_formatter.format(epoch=epoch,
                                        train_loss=" ".join(["{}".format('%.3f' % train_loss)]),
                                        train_auc=" ".join(["{}".format('%.3f' % train_auc)]),
                                        train_acc=" ".join(["{}".format('%.3f' % train_acc)]),
                                        train_f1_ma=" ".join(["{}".format('%.3f' % train_f1_macro)]),
                                        train_f1_mi=" ".join(["{}".format('%.3f' % train_f1_micro)]),
                                        train_auc_c=train_auc_class,
                                        # epoch=epoch,
                                        valid_loss=" ".join(["{}".format('%.3f' % val_loss)]),
                                        valid_auc=" ".join(["{}".format('%.3f' % val_auc)]),
                                        valid_acc=" ".join(["{}".format('%.3f' % val_acc)]),
                                        valid_f1_ma=" ".join(["{}".format('%.3f' % val_f1_macro)]),
                                        valid_f1_mi=" ".join(["{}".format('%.3f' % val_f1_micro)]),
                                        valid_auc_c=val_auc_class,)
    save_root.mkdir(exist_ok=True)
    weight_save_path.mkdir(exist_ok=True)

    with open(log_filepath, "a") as f:
        f.write(to_write)

    if val_loss < best_loss:
        best_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), os.path.join(weight_save_path, 'model-{}.pth').format(epoch))
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement == patience:
        print('Early stopping at epoch {}...'.format(epoch+1))
        torch.save(model.state_dict(), os.path.join(weight_save_path, 'model-{}.pth').format(epoch))
        break

test_root = pathlib.Path('/mnt/hd0/original_datasets/JT_Breast/SUQH/40xTiles')
test_split = pathlib.Path('/mnt/hd0/project_large_files/bca_grading/suqh/split_data/test.txt')

weights_dir = weight_save_path
logs_save_path = weights_dir

log_filepath.parent.mkdir(exist_ok=True)
copy_self(save_root/log_time/'run.py')

if os.path.exists(logs_save_path) is False:
    os.makedirs(logs_save_path)
log_filename = 'test_'+time.strftime("%Y_%m_%d_%H:%M:%S") + '.txt'
# log_filename = 'val_'+time.strftime("%Y_%m_%d_%H:%M:%S") + '.txt'
log_filepath = os.path.join(logs_save_path, log_filename)

img_size = 224

# 实例化test数据集
test_dataset = MyDataSet(data_root=test_root,
                            split_file=test_split,
                            transform=data_trans['test'])
batch_size = 16
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=nw,
                                            collate_fn=test_dataset.collate_fn)

# create model
# model = create_model(num_classes=2).to(device)
# load model weights

for weights_path in weights_dir.rglob("*.pth"):
    # print(weights_path)
    # continue

    model_weight_path = weights_path
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    # validate
    test_loss, test_auc, test_acc, f1_macro, qwk, auc_class, p_class, r_class, f_class = test(model=model,
                                                                                                    data_loader=test_loader,
                                                                                                    device=device,
                                                                                                    weight_name=weights_path.name)

    # write to txt
    log_txt_formatter = "{weights_n}: [AUC] {test_auc} [ACC] {test_acc} [f1_macro] {test_f1_ma} [Q-wK] {qwk}" \
                        " [AUC] {auc_c} [P] {p_c} [R] {r_c} [F] {f_c}\n"

    to_write = log_txt_formatter.format(weights_n="{: <12}".format(weights_path.name),
                                        test_auc=" ".join(["{}".format('%.3f' % test_auc)]),
                                        test_acc=" ".join(["{}".format('%.3f' % test_acc)]),
                                        test_f1_ma=" ".join(["{}".format('%.3f' % f1_macro)]),
                                        qwk=" ".join(["{}".format('%.3f' % qwk)]),
                                        auc_c=auc_class,
                                        p_c=p_class,
                                        r_c=r_class,
                                        f_c=f_class,)
    print(to_write)
    with open(log_filepath, "a") as f:
        f.write(to_write)

