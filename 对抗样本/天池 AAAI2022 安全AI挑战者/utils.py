import torch
import os
from models import *
import torch.nn.functional as F
# import torchvision
import numpy as np


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_model(arch):
    model = globals()[arch]()
    model.eval()
    return model


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AssembleModel(torch.nn.Module):
    def __init__(self, model1, model2):
        super(AssembleModel, self).__init__()
        assert isinstance(model1, torch.nn.Module)
        self.model1 = model1
        self.model2 = model2
        self.debug=True

    def forward(self, x):
        pred1=self.model1(x)
        pred2=self.model2(x)

        if self.debug:
            print(pred1)
            print(pred2)
            print(pred1 * pred2)
            # self.debug=False
        return pred1*pred2
