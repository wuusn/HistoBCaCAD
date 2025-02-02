import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F


def img_match(ncl_dir,im_path):
    filepath, fullname = os.path.split(im_path)
    ncl_path = os.path.join(ncl_dir,fullname[:2],fullname)
    return ncl_path


def read_split_data(root: str, ncl: str, epi: str, tub: str, mit: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    train_images_ncl = []
    val_images_ncl = []
    train_images_epi = []
    val_images_epi = []
    train_images_tub = []
    val_images_tub = []
    train_images_mit = []
    val_images_mit = []
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG",".tif"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    # for cla in flower_class:
    #     cla_path = os.path.join(root, cla)
    #     # 遍历获取supported支持的所有文件路径
    #     images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
    #               if os.path.splitext(i)[-1] in supported]
    #     # 获取该类别对应的索引
    #     image_class = class_indices[cla]
    #     # 记录该类别的样本数量
    #     every_class_num.append(len(images))
    #     # 按比例随机采样验证样本
    #     val_path = random.sample(images, k=int(len(images) * val_rate))
    #
    #     for img_path in images:
    #         if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
    #             val_images_path.append(img_path)
    #             val_images_label.append(image_class)
    #             val_images_ncl.append(img_match(ncl,img_path))
    #             val_images_epi.append(img_match(epi, img_path))
    #             val_images_tub.append(img_match(tub, img_path))
    #             val_images_mit.append(img_match(mit, img_path))
    #         else:  # 否则存入训练集
    #             train_images_path.append(img_path)
    #             train_images_label.append(image_class)
    #             train_images_ncl.append(img_match(ncl,img_path))
    #             train_images_epi.append(img_match(epi, img_path))
    #             train_images_tub.append(img_match(tub, img_path))
    #             train_images_mit.append(img_match(mit, img_path))

    # ------------------4-26--------------------------
    for cla in flower_class:
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        cla_path = os.path.join(root, cla)
        img_dirs = [os.path.join(cla_path, img_dir) for img_dir in os.listdir(cla_path)]
        val_dirs = random.sample(img_dirs, k=int(len(img_dirs) * val_rate))
        for index, img_d in enumerate(img_dirs):
            images = [os.path.join(img_d, i) for i in os.listdir(img_d) if os.path.splitext(i)[-1] in supported]
            every_class_num.append(len(images))
            for img_path in images:
                if os.path.split(img_path)[0] in val_dirs:
                    val_images_path.append(img_path)
                    val_images_label.append(image_class)
                    val_images_ncl.append(img_match(ncl,img_path))
                    val_images_epi.append(img_match(epi, img_path))
                    val_images_tub.append(img_match(tub, img_path))
                    val_images_mit.append(img_match(mit, img_path))
                else:  # 否则存入训练集
                    train_images_path.append(img_path)
                    train_images_label.append(image_class)
                    train_images_ncl.append(img_match(ncl,img_path))
                    train_images_epi.append(img_match(epi, img_path))
                    train_images_tub.append(img_match(tub, img_path))
                    train_images_mit.append(img_match(mit, img_path))
    # ------------------4-26--------------------------

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, train_images_ncl, train_images_epi, train_images_tub, train_images_mit,\
           val_images_path, val_images_label, val_images_ncl, val_images_epi, val_images_tub, val_images_mit


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


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

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
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
        # scheduler.step()

    y_true2 = label_binarize(y_true, classes=[0, 1, 2])
    auc_class = roc_auc_score(y_true2, y_score_auc, multi_class="ovo", average=None)
    auc1 = roc_auc_score(y_true2, y_score_auc, multi_class="ovo", average="macro")

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, auc1, f1_macro, f1_micro, auc_class


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

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
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

    y_true2 = label_binarize(y_true, classes=[0, 1, 2])
    auc_class = roc_auc_score(y_true2, y_score_auc, multi_class="ovo", average=None)
    auc1 = roc_auc_score(y_true2, y_score_auc, multi_class="ovo", average="macro")

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, auc1, f1_macro, f1_micro, auc_class
