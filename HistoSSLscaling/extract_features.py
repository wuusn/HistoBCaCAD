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


device = "cuda:0" if torch.cuda.is_available() else "cpu"
weights_path = '/home/yuxin/Downloads/ibot_vit_base_pancan.pth'
ibot_base_pancancer = iBOTViT(architecture="vit_base_pancan", encoder="student", weights_path=weights_path).to(device)

patch_size = 224
data_root = pathlib.Path('/mnt/hd0/original_datasets/JT_Breast/SUQH/40xTiles')
split_root = pathlib.Path('/mnt/hd0/project_large_files/bca_grading/suqh/split_data')
phases = ['train', 'val', 'test']

data_trans = transforms.Compose([
                                transforms.CenterCrop(patch_size*4),
                                transforms.Resize(patch_size),
                                ibot_base_pancancer.transform,
                                ])
# finetune_model_path = pathlib.Path('/mnt/hd0/project_large_files/bca_grading/suqh/histosslscaling_finetune/full_2023_10_24_10:42:58/model-0.pth')
finetune_model_path = pathlib.Path('/mnt/hd0/project_large_files/bca_grading/suqh/histosslscaling_finetune/2023_10_25_16:21:16_AUC_0.73/model-0.pth')
save_root = finetune_model_path.parent / f'center_crop_features_10x_{patch_size}'
save_root.mkdir(exist_ok=True)

num_classes= 2
ibot_base_pancancer.head = torch.nn.Linear(768, num_classes)
ibot_base_pancancer.load_state_dict(torch.load(finetune_model_path, map_location=device))


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, data_root:str, split_file:str,transform):
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
        img = self.transform(img)

        return img, label, self.images_path[item]

def post_process(feature, label, path, phase):
    path = pathlib.Path(path)
    savepath = save_root / phase / str(label) / f'{path.stem}.npy'
    savepath.parent.mkdir(exist_ok=True, parents=True)
    np.save(savepath, feature)

bs = 16
n_worker = 16
for phase in phases:
    split_file = split_root / f'{phase}.txt'
    dataset = MyDataSet(data_root=data_root,split_file=split_file, transform=data_trans)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=n_worker)

    for batch in tqdm(dataloader, desc=f'Processing {phase}'):
        imgs, labels, paths = batch
        imgs = imgs.to(device)
        features = ibot_base_pancancer(imgs).detach().cpu().numpy()
        labels = labels.numpy()
        
        with Pool(n_worker) as p:
            p.starmap(post_process, zip(features, labels, paths, [phase]*len(features)))
