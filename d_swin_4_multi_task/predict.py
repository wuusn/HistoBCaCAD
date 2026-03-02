import os
import sys
import json
import pickle
import random
import time
from PIL import Image

import torch
from tqdm import tqdm
from torchvision import transforms
from model import swin_tiny_patch4_window7_224 as create_model
from my_dataset import MyDataSet
import torch.nn.functional as F

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,precision_recall_fscore_support,cohen_kappa_score,roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
Image.MAX_IMAGE_PIXELS = None

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def evaluate(model, data_loader, device, weight_name):

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

        pred = model(images.to(device))
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
    y_true2 = label_binarize(y_true, classes=[0, 1, 2])
    auc_class = roc_auc_score(y_true2, y_score_auc, multi_class="ovo", average=None)
    auc1 = roc_auc_score(y_true2, y_score_auc, multi_class="ovo", average="macro")

    # test_loss, test_auc, test_acc, f1_macro, qwk, auc_class, p_class, r_class, f_class
    return accu_loss.item() / (step + 1), auc1, f1_micro, f1_macro, qwk, auc_class, p_class, r_class, f_class


def img_match(ncl_dir,im_path):
    filepath, fullname = os.path.split(im_path)
    ncl_path = os.path.join(ncl_dir, fullname)
    return ncl_path


if __name__=='__main__':
    logs_save_path = '/mnt/hd0/project_large_files/bca_grading/suqh/baibai_mod/logs'
    test_root = '/mnt/hd0/original_datasets/JT_Breast/SUQH/40xTiles'
    test_split = '/mnt/hd0/project_large_files/bca_grading/suqh/split_data/test.txt'
    # test_split = '/mnt/hd0/project_large_files/bca_grading/suqh/split_data/val.txt'
    # test_path = '/mnt/disk1/bjy/dataset/Biox/test/'
    # ncl_dir = '/mnt/disk1/bjy/dataset/Biox/ncl/ori/test/'
    # epi_dir = '/mnt/disk1/bjy/dataset/Biox/ncl/epi/test/'
    # tub_dir = '/mnt/disk1/bjy/dataset/Biox/ncl/tub/test/'
    # mit_dir = '/mnt/disk1/bjy/dataset/Biox/ncl/mit/test/'
    weights_dir = '/mnt/hd0/project_large_files/bca_grading/suqh/baibai_mod/weights'

    if os.path.exists(logs_save_path) is False:
        os.makedirs(logs_save_path)
    log_filename = 'test_'+time.strftime("%Y_%m_%d_%H:%M:%S") + '.txt'
    # log_filename = 'val_'+time.strftime("%Y_%m_%d_%H:%M:%S") + '.txt'
    log_filepath = os.path.join(logs_save_path, log_filename)

    # test_list=os.listdir(test_path)
    # assert os.path.exists(test_path), "dataset root: {} does not exist.".format(test_path)

    # test_images_path = []  # 存储test的所有图片路径
    # test_images_label = []  # 存储test图片对应索引信息
    # test_images_ncl = []
    # test_images_epi = []
    # test_images_tub = []
    # test_images_mit = []

    # for index,img_name in enumerate(test_list):
    #     # print(img_name)
    #     img_path=os.path.join(test_path,img_name)
    #     # print(img_path)
    #     test_images_path.append(img_path)
    #     test_images_ncl.append(img_match(ncl_dir, img_path))
    #     test_images_epi.append(img_match(epi_dir, img_path))
    #     test_images_tub.append(img_match(tub_dir, img_path))
    #     test_images_mit.append(img_match(mit_dir, img_path))
    #     # read class_indict
    #     json_path = './class_indices.json'
    #     assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    #     with open(json_path, "r") as f:
    #         class_indict = json.load(f)
    #     key= int([k for k,v in class_indict.items() if v==img_name[:2]][0])
    #     test_images_label.append(key)
    # print(test_images_path)
    # print(test_images_label)

    img_size = 224
    data_transform = transforms.Compose(
        [
        #  transforms.Resize(int(img_size * 1.14)),
        #  transforms.CenterCrop(img_size),
         transforms.CenterCrop(img_size*4),
         transforms.Resize(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 实例化test数据集
    test_dataset = MyDataSet(data_root=test_root,
                             split_file=test_split,
                             transform=data_transform)
    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=nw,
                                              collate_fn=test_dataset.collate_fn)

    # create model
    model = create_model(num_classes=3).to(device)
    # load model weights
    weight_lists=os.listdir(weights_dir)
    for w in weight_lists:
        weights_path=os.path.join(weights_dir,w)

        model_weight_path = weights_path
        model.load_state_dict(torch.load(model_weight_path, map_location=device))

        # validate
        test_loss, test_auc, test_acc, f1_macro, qwk, auc_class, p_class, r_class, f_class = evaluate(model=model,
                                                                                                      data_loader=test_loader,
                                                                                                      device=device,
                                                                                                      weight_name=w)

        # write to txt
        log_txt_formatter = "{weights_n}: [AUC] {test_auc} [ACC] {test_acc} [f1_macro] {test_f1_ma} [Q-wK] {qwk}" \
                            " [AUC] {auc_c} [P] {p_c} [R] {r_c} [F] {f_c}\n"

        to_write = log_txt_formatter.format(weights_n="{: <12}".format(w),
                                            test_auc=" ".join(["{}".format('%.3f' % test_auc)]),
                                            test_acc=" ".join(["{}".format('%.3f' % test_acc)]),
                                            test_f1_ma=" ".join(["{}".format('%.3f' % f1_macro)]),
                                            qwk=" ".join(["{}".format('%.3f' % qwk)]),
                                            auc_c=auc_class,
                                            p_c=p_class,
                                            r_c=r_class,
                                            f_c=f_class,)
        with open(log_filepath, "a") as f:
            f.write(to_write)



