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
from config import args_resnet, args_densenet
from utils import load_model, AverageMeter, accuracy

import argparse
from tqdm import tqdm
import torchvision
import torch.nn as nn
# 新增：
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sys

# Use CUDA
use_cuda = torch.cuda.is_available()

seed = 11037
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_name, label_name, transform):
        images = np.load(data_name)
        labels = np.load(label_name)
        assert labels.min() >= 0
        assert images.dtype == np.uint8
        assert images.shape[0] <= 50000
        assert images.shape[1:] == (32, 32, 3)
        self.images = [Image.fromarray(x) for x in images]
        self.labels = labels / labels.sum(axis=1, keepdims=True)  # normalize
        self.labels = self.labels.astype(np.float32)
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)


def cross_entropy(outputs, smooth_labels):
    loss = torch.nn.KLDivLoss(reduction='batchmean')
    return loss(F.log_softmax(outputs, dim=1), smooth_labels)


def main():
    for arch in ['resnet50','densenet121']:
        if arch == 'resnet50':
            args = args_resnet
        else:
            args = args_densenet
        assert args['epochs'] <= 200
        # Data
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = MyDataset(data_name='./2wtest_3wFGSM_images.npy', label_name='./2wtest_3wFGSM_labels.npy', transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=True, num_workers=0)
        testset = MyDataset(data_name='./cifar_image_test.npy', label_name='./cifar_label_test.npy', transform=transform_train)
        testloader = data.DataLoader(testset, batch_size=2 * args['batch_size'], shuffle=True, num_workers=0)
        # Model

        model = load_model(arch)
        best_acc = 0  # best test accuracy

        optimizer = optim.__dict__[args['optimizer_name']](model.parameters(),
                                                           **args['optimizer_hyperparameters'])
        if args['scheduler_name'] != None:
            scheduler = torch.optim.lr_scheduler.__dict__[args['scheduler_name']](optimizer,
                                                                                  **args['scheduler_hyperparameters'])
        model = model.cuda()
        # Train and val
        for epoch in range(args['epochs']):

            train_loss, train_acc = train(trainloader, model, optimizer, epoch)
            test_loss, test_acc = test(testloader, model, epoch)

            print('acc: {}'.format(train_acc))

            # save model
            if best_acc<test_acc:
                best_acc = max(test_acc, best_acc)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }, arch=arch)
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
    for steps, (inputs, soft_labels) in bar:
        inputs, soft_labels = inputs.cuda(), soft_labels.cuda()
        targets = soft_labels.argmax(dim=1)
        outputs = model(inputs)
        loss = cross_entropy(outputs, soft_labels)
        acc = accuracy(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), inputs.size(0))
        accs.update(acc[0].item(), inputs.size(0))

        bar.set_postfix(Epoch=epoch,Train_loss=losses.avg,Train_acc=accs.avg,LR=optimizer.param_groups[0]['lr'])
    return losses.avg, accs.avg


@torch.no_grad()
def test(testloader, model,epoch):
    losses = AverageMeter()
    accs = AverageMeter()
    model.eval()

    bar = tqdm(enumerate(testloader), total=len(testloader))
    for steps, (inputs, soft_labels) in bar:
        inputs, soft_labels = inputs.cuda(), soft_labels.cuda()
        targets = soft_labels.argmax(dim=1)
        outputs = model(inputs)
        loss = cross_entropy(outputs, soft_labels)
        acc = accuracy(outputs, targets)

        losses.update(loss.item(), inputs.size(0))
        accs.update(acc[0].item(), inputs.size(0))

        bar.set_postfix(Epoch=epoch, Test_loss=losses.avg,Test_acc=accs.avg)
    return losses.avg, accs.avg


def save_checkpoint(state, arch):
    filepath = os.path.join(arch + '.pth.tar')
    torch.save(state, filepath)


if __name__ == '__main__':
    main()
