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

import os
CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES').split(',')
CUDA_VISIBLE_DEVICES=[int(i) for i in CUDA_VISIBLE_DEVICES]
print("****",type(CUDA_VISIBLE_DEVICES),len(CUDA_VISIBLE_DEVICES),CUDA_VISIBLE_DEVICES)
print("^^^^",torch.cuda.get_device_name(0))
print("^^^^",torch.cuda.get_device_name(1))

### 初始化我们的模型、数据、各种配置  ####
# DDP：从外部得到local_rank参数
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
FLAGS = parser.parse_args()
local_rank = FLAGS.local_rank

# DDP：DDP backend初始化
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

# Use CUDA
use_cuda = torch.cuda.is_available()

seed = 11037
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


transform = torchvision.transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

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
    loss = torch.nn.KLDivLoss(reduction='batchmean').to(local_rank)
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

        # Model
        # 构造模型
        model = load_model(arch)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(local_rank)
        # DDP: 构造DDP model
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)



        best_acc = 0  # best test accuracy

        trainset = MyDataset(data_name='./cifar_image_train.npy', label_name='./cifar_label_train.npy',
                             transform=transform)
        testset = MyDataset(data_name='./cifar_image_test.npy', label_name='./cifar_label_test.npy',
                            transform=transform)

        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset)

        trainloader = data.DataLoader(trainset, batch_size=args['batch_size'], num_workers=4, sampler=train_sampler)
        testloader = data.DataLoader(testset, batch_size=2 * args['batch_size'],  num_workers=4, sampler=test_sampler)

        optimizer = optim.__dict__[args['optimizer_name']](model.parameters(),
                                                           **args['optimizer_hyperparameters'])
        if args['scheduler_name'] != None:
            scheduler = torch.optim.lr_scheduler.__dict__[args['scheduler_name']](optimizer,
                                                                                  **args['scheduler_hyperparameters'])
        # Train and val
        for epoch in range(args['epochs']):
            trainloader.sampler.set_epoch(epoch)
            testloader.sampler.set_epoch(epoch)

            train_loss, train_acc = train(trainloader, model, optimizer, epoch)
            test_loss, test_acc = test(testloader, model, epoch)

            print('acc: {}'.format(train_acc))

            # save model
            if dist.get_rank() == 0:
                best_acc = max(test_acc, best_acc)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.module.state_dict(),
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
        inputs, soft_labels = inputs.to(local_rank), soft_labels.to(local_rank)
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
        inputs, soft_labels = inputs.to(local_rank), soft_labels.to(local_rank)

        targets = soft_labels.argmax(dim=1)
        outputs = model(inputs)
        loss = cross_entropy(outputs, soft_labels)
        acc = accuracy(outputs, targets)

        losses.update(loss.item(), inputs.size(0))
        accs.update(acc[0].item(), inputs.size(0))

        bar.set_postfix(Epoch=epoch, Test_loss=losses.avg,Test_acc=accs.avg)
    return losses.avg, accs.avg


def save_checkpoint(state, arch):
    filepath = os.path.join(arch + '_average.pth.tar')
    torch.save(state, filepath)


if __name__ == '__main__':
    main()
