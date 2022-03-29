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
import cv2

from config import cfg
from dataset import *
from utils import *
from attack import *

# Use CUDA
use_cuda = torch.cuda.is_available()

seed = 11037
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def main():
    args = cfg

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

        if cfg['attack']:
            attacked_inputs = attack(inputs, model, params)
            attacked_outputs = model(attacked_inputs)
            attacked_loss = cross_entropy(attacked_outputs, labels)
            loss += attacked_loss

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
