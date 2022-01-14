from deeprobust.image.attack.pgd import PGD
from deeprobust.image.attack.deepfool import DeepFool
# from deeprobust.image.config import attack_params
from attack_params import *
import torch
import deeprobust.image.netmodels.resnet as resnet
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import random
from PIL import Image
from utils import load_model, AverageMeter, accuracy
from attack import *
import copy
cifar_image_train = np.load('./cifar_image_train.npy').astype(np.float64)
PGD_data = np.load('./PGD_data.npy').astype(np.float64)
FGSM_data = np.load('./FGSM_data.npy').astype(np.float64)


print(cifar_image_train[0][0][0])
print(PGD_data[0][0][0])
print(FGSM_data[0][0][0])
result=(cifar_image_train+PGD_data+FGSM_data)/3
print(result[0][0][0])


# a = np.load("PGD_label.npy")
# result_image = np.argmax(a, axis=1)
# temp = [np.argwhere(result_image == i) for i in range(10)]


# result=copy.deepcopy(cifar_image_train)
# for i in range(10):
#     instance_index=temp[i]
#
#     instance_index1=instance_index[:1000].squeeze()
#     result[instance_index1]=PGD_data[instance_index1]
#
#     instance_index2 = instance_index[1000:2000].squeeze()
#     result[instance_index2] = FGSM_data[instance_index2]
# # s/s

result=result.astype('uint8')

# for i in range(result.shape[0]):
#     img = Image.fromarray(result[i]).convert('RGB')
#     img.show()
np.save('attack_1_FGSM_PGD_average.npy',result)




# print(a[[0,1],[1,6]])


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
        self.labels = self.labels.astype(np.longlong)
        self.labels = torch.tensor([x[1] for x in np.argwhere(self.labels == 1)])
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)
