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
from sklearn.metrics import f1_score,precision_recall_fscore_support,cohen_kappa_score,roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F
import sys
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

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
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        # img_full, img_name = os.path.split(self.images_path[item])
        # img_n, ext = os.path.splitext(img_name)

        seed = np.random.randint(2147483647)
        if self.transform is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            img = self.transform(img)


        # img.save(os.path.join('./img_out',img_n+'_1'+'.jpg'))
        # ncl.save(os.path.join('./img_out', img_n + '_2' + '.jpg'))
        # epi.save(os.path.join('./img_out', img_n + '_3' + '.jpg'))
        # tub.save(os.path.join('./img_out', img_n + '_4' + '.jpg'))
        # mit.save(os.path.join('./img_out', img_n + '_5' + '.jpg'))

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)

        return images, labels

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
    y_true2 = label_binarize(y_true, classes=[0, 1, 2])
    auc_class = roc_auc_score(y_true2, y_score_auc, multi_class="ovo", average=None)
    auc1 = roc_auc_score(y_true2, y_score_auc, multi_class="ovo", average="macro")
    # auc_score = roc_auc_score(y_true, y_score_auc[:,1])
    # auc1 = auc_class = auc_score

    # test_loss, test_auc, test_acc, f1_macro, qwk, auc_class, p_class, r_class, f_class
    return accu_loss.item() / (step + 1), auc1, f1_micro, f1_macro, qwk, auc_class, p_class, r_class, f_class


def create_model(num_classes=2):
    weights_path = '/home/yuxin/Downloads/ibot_vit_base_pancan.pth'
    ibot_base_pancancer = iBOTViT(architecture="vit_base_pancan", encoder="student", weights_path=weights_path)
    ibot_base_pancancer.head = torch.nn.Linear(768, num_classes)

    return ibot_base_pancancer


if __name__=='__main__':
    save_root = pathlib.Path('/mnt/hd0/project_large_files/bca_grading/suqh/histosslscaling_finetune/full_N3_2023_10_24_16:09:46')
    logs_save_path = save_root
    test_root = pathlib.Path('/mnt/hd0/original_datasets/JT_Breast/SUQH/40xTiles')
    test_split = pathlib.Path('/mnt/hd0/project_large_files/bca_grading/suqh/split_data/test.txt')
    weights_dir = save_root

    if os.path.exists(logs_save_path) is False:
        os.makedirs(logs_save_path)
    log_filename = 'test_'+time.strftime("%Y_%m_%d_%H:%M:%S") + '.txt'
    # log_filename = 'val_'+time.strftime("%Y_%m_%d_%H:%M:%S") + '.txt'
    log_filepath = os.path.join(logs_save_path, log_filename)

    img_size = 224
    data_transform = transforms.Compose(
        [
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
    
    for weights_path in weights_dir.rglob("*.pth"):
        # print(weights_path)
        # continue

        model_weight_path = weights_path
        model.load_state_dict(torch.load(model_weight_path, map_location=device))

        # validate
        test_loss, test_auc, test_acc, f1_macro, qwk, auc_class, p_class, r_class, f_class = evaluate(model=model,
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
        with open(log_filepath, "a") as f:
            f.write(to_write)



