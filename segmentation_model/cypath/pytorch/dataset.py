from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
from PIL import Image
from .normalizeStaining import normalizeStaining
import random


class PatchDataL2(Dataset):

    def __init__(self, base_dir, base_ext, label2N, dp_transforms=None, dl_transforms=None, params=None):
        self.label2N = label2N
        self.dl_transforms = dl_transforms
        self.dp_transforms = dp_transforms
        self.params = params
        self.base_dir = base_dir
        self.base_ext = base_ext
        self.paths = []
        if self.params is not None and self.params.get('balance'):
            self.min_size = float('inf')
            for k in self.label2N.keys():
                tmp = glob.glob(f'{base_dir}/{k}/*{base_ext}')
                size = len(tmp)
                self.min_size = size if self.min_size > size else self.min_size

            for k in self.label2N.keys():
                tmp = glob.glob(f'{base_dir}/{k}/*{base_ext}')
                size = len(tmp)
                if size > self.min_size:
                    random.shuffle(tmp)
                    self.paths.extend(tmp[:self.min_size])
                else:
                    self.paths.extend(tmp)
        else:
            for k in self.label2N.keys():
                tmp = glob.glob(f'{base_dir}/{k}/*{base_ext}')
                self.paths.extend(tmp)


    def reset_paths(self):
        if self.params is not None and self.params.get('balance'):
            self.paths = []
            for k in self.label2N.keys():
                tmp = glob.glob(f'{self.base_dir}/{k}/*{self.base_ext}')
                size = len(tmp)
                if size > self.min_size:
                    random.shuffle(tmp)
                    self.paths.extend(tmp[:self.min_size])
                else:
                    self.paths.extend(tmp)


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.get_label(path)
        im = Image.open(path)
        ori_im = im.copy()

        if self.dp_transforms:
            for transform in self.dp_transforms:
                ftran = getattr(self, transform+'_transform')
                im = ftran(im, self.params)

        if self.dl_transforms:
            transforms = self.dl_transforms.transforms
            for t in transforms[:-2]:
                im = t(im)

        new_im = im.copy()

        if self.dl_transforms:
            for t in [transforms[-2], transforms[-1]]:
                im = t(im)
        else:
            im = self.im2tensor(im)

        tsr_im = im
        ori_im = np.array(ori_im)
        new_im = np.array(new_im)

        return {'image': tsr_im, 'label': label, 'ori_im': ori_im, 'new_im': new_im}

    def get_label(self, path):
        label_name = path.split('/')[-2]
        label = self.label2N[label_name]
        return label

    def macenko_transform(self, im, params):
        np_im = np.array(im)
        np_im.astype(np.float128)
        np_im, h, e  = normalizeStaining(np_im)
        im = Image.fromarray(np_im)
        return im

    def im2tensor(self, im):
        np_im = np.array(im)
        np_im = np_im.transpose((2,0,1))
        np_im = np_im / 255
        tsr_im = torch.from_numpy(np_im)
        return tsr_im.float()

    def flip_transform(self, im):
        if random() > .5:
            im = TF.hflip(im)
        if random() > .5:
            im = TF.vflip(im)
        return im

    def simple_color_transform(self, im):
        if random() > .5:
            im = TF.adjust_brightness(im, uniform(1, 1.4))
        if random() > .5:
            im = TF.adjust_contrast(im, uniform(1, 1.4))
        if random() > .5:
            im = TF.adjust_saturation(im, uniform(1, 1.4))
        if random() > .5:
            im = TF.adjust_hue(im, uniform(-.5, .5))
        if random() > .5:
            im = im.filter(ImageFilter.GaussianBlur(int(random()>.5)+1))
        return im

class ClassificationModelData(PatchDataL2):

    def __init__(self, base_dir, base_ext, label2N, dp_transforms=None, dl_transforms=None, params=None):
        super().__init__(base_dir, base_ext, label2N, dp_transforms, dl_transforms, params)

class ClassificationTestData(ClassificationModelData):
    def __init__(self, base_dir, base_ext, label2N, dp_transforms=None, dl_transforms=None, params=None):
        paths1 = glob.glob(f'{base_dir}/normal/*/*{base_ext}')
        paths2 = glob.glob(f'{base_dir}/tumor/*/*/*{base_ext}')
        paths1.extend(paths2)
        self.base_dir = base_dir
        self.paths = paths1
        self.label2N = label2N
        self.dl_transforms = dl_transforms
        self.dp_transforms = dp_transforms
        self.params = params

    def __getitem__(self, index):
        path = self.paths[index]
        tmp_path = path.replace(self.base_dir, '')
        if 'normal' in tmp_path:
            label_name = 'normal'
        else:
            label_name = path.split('/')[-2].split('-')[0]

        label = self.label2N[label_name]
        im = Image.open(path)
        ori_im = im.copy()

        if self.dp_transforms:
            for transform in self.dp_transforms:
                ftran = getattr(self, transform+'_transform')
                im = ftran(im, self.params)

        if self.dl_transforms:
            transforms = self.dl_transforms.transforms
            for t in transforms[:-2]:
                im = t(im)

        new_im = im.copy()

        if self.dl_transforms:
            for t in [transforms[-2], transforms[-1]]:
                im = t(im)
        else:
            im = self.im2tensor(im)

        tsr_im = im
        ori_im = np.array(ori_im)
        new_im = np.array(new_im)

        return {'image': tsr_im, 'label': label, 'ori_im': ori_im, 'new_im': new_im}

class ClassificationOutlierData(ClassificationModelData):
    def __init__(self, base_dir, base_ext, label2N, dp_transforms=None, dl_transforms=None, params=None):
        paths = glob.glob(f'{base_dir}/*/*/*{base_ext}')
        self.base_dir = base_dir
        self.paths = paths
        self.label2N = label2N
        self.dl_transforms = dl_transforms
        self.dp_transforms = dp_transforms
        self.params = params

    def __getitem__(self, index):
        path = self.paths[index]
        tmp_path = path.replace(self.base_dir, '')
        label_name = path.split('/')[-2].split('-')[0]

        label = self.label2N[label_name]
        im = Image.open(path)
        ori_im = im.copy()

        if self.dp_transforms:
            for transform in self.dp_transforms:
                ftran = getattr(self, transform+'_transform')
                im = ftran(im, self.params)

        if self.dl_transforms:
            transforms = self.dl_transforms.transforms
            for t in transforms[:-2]:
                im = t(im)

        new_im = im.copy()

        if self.dl_transforms:
            for t in [transforms[-2], transforms[-1]]:
                im = t(im)
        else:
            im = self.im2tensor(im)

        tsr_im = im
        ori_im = np.array(ori_im)
        new_im = np.array(new_im)

        return {'image': tsr_im, 'label': label, 'ori_im': ori_im, 'new_im': new_im}

class RegionDataL2(PatchDataL2):
    def __init__(self, base_dir, region_ext, patch_ext, label2N, dp_transforms=None, dl_transforms=None, params=None):
        self.region_ext = region_ext
        self.patch_ext = patch_ext
        super().__init__(base_dir, region_ext, label2N, dp_transforms, dl_transforms, params)

    def __getitem__(self, index):
        region_path = self.paths[index]
        image_list = []
        label = None
        paths = glob.glob(f'{region_path}/*{self.patch_ext}')
        region_size = len(paths)

        batch_tsr_im = None
        batch_ori_im = None
        batch_new_im = None

        for path in  paths:
            if label==None:
                label_name = path.split('/')[-2]
                label = self.label2N[label_name]

            im = Image.open(path)
            ori_im = im.copy()

            if self.dp_transforms:
                for transform in self.dp_transforms:
                    ftran = getattr(self, transform+'_transform')
                    im = ftran(im, self.params)

            if self.dl_transforms:
                transforms = self.dl_transforms.transforms
                for t in transforms[:-2]:
                    im = t(im)

            new_im = im.copy()

            if self.dl_transforms:
                for t in [transforms[-2], transforms[-1]]:
                    im = t(im)
            else:
                im = self.im2tensor(im)

            if batch_tsr_im is None:
                _np_arr = np.array(ori_im)
                ori_im = np.expand_dims(_np_arr, axis=0)
                _np_arr = np.array(new_im)
                new_im = np.expand_dims(_np_arr, axis=0)

                batch_tsr_im = im.unsqueeze(0)
                batch_ori_im = ori_im
                batch_new_im = new_im
            else:
                _np_arr = np.array(ori_im)
                ori_im = np.expand_dims(_np_arr, axis=0)
                _np_arr = np.array(new_im)
                new_im = np.expand_dims(_np_arr, axis=0)

                batch_tsr_im = torch.cat((batch_tsr_im, im.unsqueeze(0)), 0)
                batch_ori_im = np.concatenate((batch_ori_im, ori_im), axis=0)
                batch_new_im = np.concatenate((batch_new_im, new_im), axis=0)



        return {'image': batch_tsr_im, 'label': label, 'ori_im': batch_ori_im, 'new_im': batch_new_im}

class PatchDataL3(PatchDataL2):
    def __init__(self, base_dir, base_ext, label2N, dp_transforms=None, dl_transforms=None, params=None):
        self.label2N = label2N
        self.dl_transforms = dl_transforms
        self.dp_transforms = dp_transforms
        self.params = params
        self.base_dir = base_dir
        self.base_ext = base_ext
        self.paths = []
        self.paths = glob.glob(f'{base_dir}/*{base_ext}')

    def get_label(self, path):
        label_name = path.split('/')[-3]
        label = self.label2N[label_name]
        return label
