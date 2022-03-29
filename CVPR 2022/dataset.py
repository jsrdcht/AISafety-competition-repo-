import os
import torch
import cv2

# Albumentations for augmentations
import albumentations as A
from albumentations import ImageOnlyTransform
from albumentations.augmentations.transforms import *
from albumentations.pytorch import ToTensorV2
import random
import numpy as np
from PIL import Image


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform, mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        if self.mode == 'train':
            self.img_paths, self.labels = self.load_filenames(self.data_dir, self.mode)
        else:
            self.img_paths, self.img_names = self.load_filenames(self.data_dir, self.mode)

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
            if self.transform and random.randint(0, 9) < 1:
                temp = AddSaltPepperNoise(0.2)
                img = temp(img)
            if self.transform: img = transform_norm(image=img)["image"]

            return img, torch.tensor(label, dtype=torch.long)
        else:
            img_path = self.img_paths[index]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.transform: img = self.transform(image=img)["image"]

            return img, self.img_names[index]

    def __len__(self):
        return len(self.img_paths)


class AddSaltPepperNoise(object):

    def __init__(self, density=0):
        self.density = density

    def __call__(self, img):
        img = np.array(img)

        # 图片转numpy
        h, w, c = img.shape
        Nd = self.density
        Sd = 1 - Nd
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd])  # 生成一个通道的mask
        mask = np.repeat(mask, c, axis=2)  # 在通道的维度复制，生成彩色的mask
        img[mask == 0] = 0  # 椒
        img[mask == 1] = 255  # 盐
        # numpy转图片
        return img


transform_train = A.Compose([
    # 形态变化
    A.RandomResizedCrop(height=224, width=224, scale=(0.5, 1.0), ),
    A.HorizontalFlip(p=0.5),

    # 色彩、对比度、亮度
    A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),

    # 随机噪声
    A.GaussNoise(p=0.5),
    A.OneOf([
        # 模糊相关操作
        A.MotionBlur(p=.75),
        A.MedianBlur(blur_limit=3, p=0.5),
        A.Blur(blur_limit=3, p=0.75),
    ], p=0.5),

    # weather aug
    A.OneOf([
        A.RandomRain(p=0.2),
        A.RandomSnow(p=0.1),
        A.RandomShadow(p=0.1),
        A.RandomFog(p=0.1),
    ], p=0.2),

    # 畸变相关操作
    A.OneOf([
        A.OpticalDistortion(p=0.2),
        A.GridDistortion(p=0.25),
        A.PiecewiseAffine(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=60, p=0.3),
        A.ElasticTransform(p=0.1),
    ], p=0.2),

    # 锐化、浮雕等操作
    A.OneOf([
        A.CLAHE(clip_limit=2),
        A.Sharpen(),
        A.Emboss(),
    ], p=0.1),

    # cropout
    A.OneOf([
        A.GridDropout(ratio=0.05, p=0.1),
        A.CoarseDropout(max_holes=5, max_height=4, max_width=4, p=0.1),
    ], p=0.1),

    # 归一化
    # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
    # ToTensorV2()
], p=1.)
transform_norm = A.Compose([
    # 归一化
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
    ToTensorV2()], p=1.0
)

transform_test = A.Compose([
    A.Resize(224, 224),

    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
    ToTensorV2()], p=1.)

# Data
# transform_train = transforms.Compose([
#     transforms.RandomResizedCrop((224, 224)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
