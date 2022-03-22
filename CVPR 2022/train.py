from __future__ import print_function

import os
import random
import shutil
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from utils import load_model, AverageMeter, accuracy
import cv2

from config import cfg

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Use CUDA
use_cuda = torch.cuda.is_available()

seed = 11037
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform, mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        if self.mode == 'train':
            self.img_paths, self.labels = self.load_filenames(self.data_dir, self.mode)
        else:
            self.img_paths, self.img_names = self.load_filenames(self.data_dir, self.mode)

        assert len(self.labels) >= 0

    def load_filenames(self, data_dir, mode):
        if mode == 'train':
            with open(os.path.join(data_dir, 'label.txt'), 'r') as f:
                data = f.readlines()
            img_paths = [os.path.join(data_dir, 'train_images', _.split()[0]) for _ in data]
            labels = [float(_.split()[-1]) for _ in data]

            return img_paths, labels
        else:
            img_paths = []
            img_names = []
            for filename in os.listdir(os.path.join(data_dir, 'test_images')):
                img_paths.append(os.path.join(data_dir, 'test_images', filename))
                img_names.append(filename)

            return img_paths, img_names

    def __getitem__(self, index):
        if self.mode == 'train':
            img_path = self.img_paths[index]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            label = self.labels[index]

            if self.transform: img = self.transform(image=img)["image"]

            return img, torch.tensor(label, dtype=torch.long)
        else:
            img_path = self.img_paths[index]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.transform: img = self.transform(image=img)["image"]

            return img, self.img_names[index]

    def __len__(self):
        return len(self.img_paths)


def cross_entropy(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)


def main():
    args = cfg
    # Data
    # transform_train = transforms.Compose([
    #     transforms.RandomResizedCrop((224, 224)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    transform_train = A.Compose([
        A.Resize(224, 224),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=60, p=0.5),
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),

        A.GaussNoise(p=0.5),
        A.OneOf([
            # 模糊相关操作
            A.MotionBlur(p=.75),
            A.MedianBlur(blur_limit=3, p=0.5),
            A.Blur(blur_limit=3, p=0.75),
        ], p=0.5),
        # ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.25),
        A.OneOf([
            # 畸变相关操作
            A.OpticalDistortion(p=0.75),
            A.GridDistortion(p=0.25),
            A.PiecewiseAffine(p=0.75),
        ], p=0.2),
        A.OneOf([
            # 锐化、浮雕等操作
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),
        ], p=0.1),
        #
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2()], p=1.)

    trainset = MyDataset(data_dir='./data', mode='train', transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=True, num_workers=0)

    # Model
    model = load_model(model_name=args['model'], pretrained=False)
    best_acc = 0  # best test accuracy

    optimizer = optim.__dict__[args['optimizer_name']](model.parameters(),
                                                       **args['optimizer_hyperparameters'])
    if args['scheduler_name'] != None:
        scheduler = torch.optim.lr_scheduler.__dict__[args['scheduler_name']](optimizer,
                                                                              **args['scheduler_hyperparameters'])
    model = model.cuda()
    # Train and val
    for epoch in range(args['epochs']):

        train_loss, train_acc = train(trainloader, model, optimizer, epoch=epoch)
        print(args)
        print('acc: {}'.format(train_acc))

        # save model
        best_acc = max(train_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': train_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, arch=args['model'] + str(best_acc))

        if args['scheduler_name'] != None:
            scheduler.step()

    print('Best acc:')
    print(best_acc)


def train(trainloader, model, optimizer, epoch):
    losses = AverageMeter()
    accs = AverageMeter()
    model.eval()
    # switch to train mode
    model.train()

    bar = tqdm(enumerate(trainloader), total=len(trainloader))
    for steps, (inputs, labels) in bar:
        labels = labels.cuda()
        inputs = inputs.to('cuda', dtype=torch.float)

        outputs = model(inputs)
        loss = cross_entropy(outputs, labels)

        acc = accuracy(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inputs.size(0))
        accs.update(acc[0].item(), inputs.size(0))

        bar.set_postfix(epoch=epoch, train_loss=losses.avg, train_acc=accs.avg,
                        lr=optimizer.state_dict()['param_groups'][0]['lr'])

    return losses.avg, accs.avg


def save_checkpoint(state, arch):
    filepath = os.path.join(arch + '.pth.tar')
    torch.save(state, filepath)


if __name__ == '__main__':
    main()
