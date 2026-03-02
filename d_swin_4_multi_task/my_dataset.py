from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import torch
from torch.utils.data import Dataset
import os
import numpy as np


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
