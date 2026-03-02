from torchvision import transforms
from apex import amp
import timm
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import time
from torch.utils.data import Dataset
from PIL import Image
import torch
from sam import SAM
import numpy as np
from cypath.pytorch.dataset import *
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import yaml

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

class MyDataset(Dataset):
    def __init__(self, phase,label_path, transform):
        self.filenames = []
        self.labels = []
        self.label_path = label_path
        #print(label_path)
        self.transform = transform
        for line in open(label_path):
            rs=line.strip()
            filename,label=rs.split(" ")
            self.filenames.append("LargeFineFoodAI/Recognition/{}".format(filename))
            self.labels.append(int(label))
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx]).convert('RGB')
        image = self.transform(image)
        return image, self.labels[idx]

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

def train_model(model, dataloaders, save, lossfunc, optimizer, scheduler, params, BS, num_epochs=10,start_epoch=-1):
    start_time = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    train_acc = []
    valid_acc = []
    writer = SummaryWriter(log_dir=os.path.join(
                    save, datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')))

    for epoch in range(start_epoch+1,start_epoch+1+num_epochs,1):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            #dataloaders[phase].dataset.reset_paths()
            torch.cuda.empty_cache()
            running_loss = 0.0
            running_corrects = 0.0
            if phase == 'train':
                model.train(True)  # Set model to training mode
                for index,data in enumerate(dataloaders[phase]):
                    inputs = data['image']
                    labels = data['label']
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    #添加cutmix
                    r = np.random.rand(1)
                    beta,cutmix_prob=0,0
                    optimizer.zero_grad()
                    if beta>0 and r<cutmix_prob and phase == 'train':
                        lam = np.random.beta(beta,beta)
                        rand_index = torch.randperm(inputs.size()[0]).cuda()
                        target_a = labels
                        target_b = labels[rand_index]
                        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                        inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                        outputs = model(inputs)
                        loss = lossfunc(outputs, target_a)*lam+lossfunc(outputs, target_b)*(1-lam)
                    else:
                        outputs = model(inputs)
                        loss = lossfunc(outputs, labels)
                    # with amp.scale_loss(loss, optimizer) as scaled_loss:
                        # scaled_loss.backward()    
                    loss.backward()
                    optimizer.first_step(zero_grad=True)
              
                    lossfunc(model(inputs),labels).backward()  # make sure to do a full forward pass
                    optimizer.second_step(zero_grad=True)    
                        
                    _, preds = torch.max(outputs.data, 1)
                    lr=optimizer.state_dict()['param_groups'][0]['lr']
                 
                    # statistics
                    running_loss += loss.data
                    #if index%50==0:
                    #    print(lr,index,len(dataloaders[phase]),loss.data.cpu().numpy())
                    running_corrects += torch.sum(preds == labels.data).to(torch.float32)
            else:
                with torch.no_grad():
                    model.eval()  
                    for index,data in enumerate(dataloaders[phase]):
                        inputs = data['image'] 
                        labels = data['label']
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                        outputs = model(inputs)
                        loss = lossfunc(outputs, labels)
                        _, preds = torch.max(outputs.data, 1)
                        lr=optimizer.state_dict()['param_groups'][0]['lr']
                 
                        running_loss += loss.data
                        #if index%50==0:
                        #    print(lr,index,len(dataloaders[phase]),loss.data.cpu().numpy())
                        running_corrects += torch.sum(preds == labels.data).to(torch.float32)
            epoch_loss = running_loss /  ((index+1)*BS)
            epoch_acc = running_corrects / ((index+1)*BS)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            writer.add_scalar(f'{phase}/Loss', epoch_loss, epoch)
            writer.add_scalar(f'{phase}/Acc', epoch_acc, epoch)

            if phase == 'val':
                valid_acc.append(epoch_acc)
            else:
                train_acc.append(epoch_acc)
            # deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                checkpoint_dict = {'epoch': epoch, 
                   'model_state_dict': model.state_dict(), 
                   'optim_state_dict': optimizer.state_dict(), 
                   'criterion_state_dict': lossfunc.state_dict()}
                torch.save(checkpoint_dict, '{}/epoch_{}.pth'.format(f'{save}/checkpoints',epoch))
                print('best') 
        # 这里使用了学习率调整策略
        #scheduler.step(valid_acc[-1])
        scheduler.step()
    elapsed_time = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        elapsed_time // 60, elapsed_time % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model, train_acc, valid_acc


def train(root, base_ext, save, model_name, ckpt_path, Epoch, img_size, BS, N_class, label2N, params):

    size = (img_size, img_size)

    image_transforms = {
        'train':
        transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val':
        transforms.Compose([
            transforms.Resize(size=size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }

    ckpt_save = f'{save}/checkpoints'
    os.makedirs(save, exist_ok=True)
    os.makedirs(ckpt_save, exist_ok=True)

    train_params = params
    val_params = None

    data = {
        'train': PatchDataL3(base_dir=f'{root}/train', base_ext=base_ext, label2N=label2N, dp_transforms=params.get('norm'), dl_transforms=image_transforms['train'], params=train_params),
        'val': PatchDataL3(base_dir=f'{root}/val', base_ext=base_ext, label2N=label2N, dp_transforms=params.get('norm'), dl_transforms=image_transforms['val'], params=val_params)
    }

    for k,v in data.items():
        print(k, len(v))

    dataloaders = {
        'train': DataLoader(data['train'], batch_size=BS, shuffle=True, pin_memory=False, num_workers=8),
        'val': DataLoader(data['val'], batch_size=BS, shuffle=False, pin_memory=False, num_workers=8),
    }

    model = timm.create_model(model_name, pretrained=params.get('pretrain', False), num_classes=N_class, pretrained_cfg_overlay=dict(file='/home/yuxin/Downloads/swin_base_patch4_window12_384_22k.pth'),)
    
    device = "cuda"
    model.to(device)

    lossfunc = nn.CrossEntropyLoss().cuda()

    base_optimizer =torch.optim.AdamW # define an optimizer for the "sharpness-aware" update
    optimizer = SAM(model.parameters(), base_optimizer,lr=params.get('lr', 0.000005), betas=(0.9, 0.999), weight_decay=params.get('weight_decay', 0.05))
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)
    model = nn.DataParallel(model)

    start_epoch = 0

    if ckpt_path:
        checkpoint = torch.load(ckpt_path)  
        model.load_state_dict(checkpoint['model_state_dict']) 
        optimizer.load_state_dict(checkpoint['optim_state_dict']) 
        lossfunc.load_state_dict(checkpoint['criterion_state_dict'])
        start_epoch = checkpoint['epoch']
        scheduler.step(start_epoch+1)
        print('load', ckpt_path)

    model, train_acc, valid_acc = train_model(model=model,
                                                 dataloaders=dataloaders,
                                                 save=save,
                                                 lossfunc=lossfunc,
                                                 optimizer=optimizer,
                                                 scheduler=scheduler,
                                                 params=params,
                                                 BS=BS,
                                                 num_epochs=Epoch,
                                                 start_epoch=start_epoch
                                            )
        
if __name__ == "__main__":
    yaml_path = sys.argv[1]
    with open(yaml_path, 'r') as f:
        param_sets = yaml.safe_load(f)
    for set_name, param in param_sets.items():
        print(set_name, param)
        root = param.get('root')
        base_ext = param.get('base_ext', '.jpg')
        codename = param.get('codename')
        save_dir = f'{root}/save/{codename}'
        model_name = param.get('model_name')
        epoch = param.get('epoch')
        img_size = param.get('img_size')
        bs = param.get('bs')
        n_class = param.get('n_class')
        label2N = param.get('label2N')
        test_cohorts = param.get('test_cohorts')
        params = param.get('params')
        ckpt_path = params.get('ckpt_path')
        train(root, base_ext, save_dir, model_name, ckpt_path, epoch, img_size, bs, n_class, label2N, params)
