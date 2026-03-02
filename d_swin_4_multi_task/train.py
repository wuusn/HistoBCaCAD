import os
import argparse
import time

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from model import swin_tiny_patch4_window7_224 as create_model
# from model import swin_large_patch4_window7_224_in22k as create_model
from utils import read_split_data, train_one_epoch, evaluate
import numpy as np
import torchvision.transforms.functional as F

class RandomPatchCrop(torch.nn.Module):
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    """

    def __init__(self, size):
        super().__init__()
        # _log_api_usage_once(self)
        self.pad_if_needed = True
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        _, height, width = F.get_dimensions(img)
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, padding_mode='symmetric')
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, padding_mode='symmetric')
        
        _, height, width = F.get_dimensions(img)
        top = np.random.randint(0, height - self.size[0]) if height > self.size[0] else 0
        left = np.random.randint(0, width - self.size[1]) if width > self.size[1] else 0
        return F.crop(img, top, left, self.size[0], self.size[1])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    weight_save_path=args.weight_save_path
    log_save_path = args.log_save_path

    if os.path.exists(weight_save_path) is False:
        os.makedirs(weight_save_path)

    if os.path.exists(log_save_path) is False:
        os.makedirs(log_save_path)
    log_filename = 'train_'+time.strftime("%Y_%m_%d_%H:%M:%S") + '.txt'
    log_filepath = os.path.join(log_save_path, log_filename)

    tb_writer = SummaryWriter()

    # # change here
    # train_images_path, train_images_label, \
    # val_images_path, val_images_label = \
    #     read_split_data(args.data_path)

    img_size = 224
    data_transform = {
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
        "val": transforms.Compose([
                                    # transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size*4),
                                   transforms.Resize(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}



    # 实例化训练数据集
    train_dataset = MyDataSet(data_root=args.data_root,
                              split_file=args.split_root+'/train.txt',
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(data_root=args.data_root,
                            split_file=args.split_root+'/val.txt',
                            transform=data_transform["val"])

    batch_size = args.batch_size
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

    model = create_model(num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)

    best_loss = np.inf
    epochs_without_improvement = 0
    patience = 7
    for epoch in range(args.epochs):
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
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-root', type=str,
                        default="/mnt/hd0/original_datasets/JT_Breast/SUQH/40xTiles")
    parser.add_argument('--split-root', type=str,
                        default="/mnt/hd0/project_large_files/bca_grading/suqh/split_data")

    # parser.add_argument('--ncl-path', type=str,
    #                     default="/mnt/disk1/bjy/dataset/Biox/ncl/ori/train_val/")

    # parser.add_argument('--epi-path', type=str,
    #                     default="/mnt/disk1/bjy/dataset/Biox/ncl/epi/train_val/")

    # parser.add_argument('--tub-path', type=str,
    #                     default="/mnt/disk1/bjy/dataset/Biox/ncl/tub/train_val/")

    # parser.add_argument('--mit-path', type=str,
    #                     default="/mnt/disk1/bjy/dataset/Biox/ncl/mit/train_val/")

    parser.add_argument('--weight-save-path', type=str,
                        default="/mnt/hd0/project_large_files/bca_grading/suqh/baibai_mod/weights")

    parser.add_argument('--log-save-path', type=str,
                        default="/mnt/hd0/project_large_files/bca_grading/suqh/baibai_mod/logs")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='/mnt/hd0/project_large_files/bca_grading/suqh/baibai/swin_tiny_patch4_window7_224.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
