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


class CFG:
    batch_size = 32


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_resnet = resnet.ResNet50()
state_resnet = torch.load("resnet50.pth.tar")
state_resnet = state_resnet['state_dict']
model_resnet.load_state_dict(state_resnet)

model_densenet = load_model('densenet121')
state_densenet = torch.load("densenet121.pth.tar")
model_state = state_densenet['state_dict']
model_densenet.load_state_dict(model_state)

transform = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
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
        self.labels = self.labels.astype(np.longlong)
        self.labels = torch.tensor([x[1] for x in np.argwhere(self.labels == 1)])
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)


data_set = MyDataset(data_name='./cifar_image_train.npy', label_name='./cifar_label_train.npy', transform=transform)
data_loader = torch.utils.data.DataLoader(data_set, batch_size=CFG.batch_size, num_workers=0)

images = []
labels = []
res_count = 0
dense_count = 0
bar = tqdm(enumerate(data_loader), total=len(data_loader))
for steps, (input, label) in bar:
    input = input.to('cuda').float()
    label = label.cuda()
    # print(label)

    model_resnet.cuda()
    model_densenet.cuda()

    # adversary = PGD(model, 'cuda')
    # Adv_img = adversary.generate(input, label, **attack_params['PGD_CIFAR10'])
    Adv_img, temp1, temp2 = pgd_attack(model_resnet, model_densenet, X=input, y=label)
    res_count += temp1
    dense_count += temp2

    Adv_img = Adv_img.detach().cpu().numpy().tolist()
    images.append(Adv_img)

    label = label.detach().cpu().numpy()
    now_batch_size = len(label)
    temp = np.zeros(now_batch_size * 10).reshape(now_batch_size, 10)
    for i in range(now_batch_size): temp[i][label[i]] = 1
    label = label.tolist()
    labels.append(temp)

    if steps >= 0: break

images, labels = np.concatenate(images, axis=0), np.concatenate(labels, axis=0)

# denormilization
images = images.transpose(1, 0, 2, 3)
print(images.shape)
images[0] = (images[0] * 0.2023 + 0.4914) * 255
images[1] = (images[1] * 0.1994 + 0.4822) * 255
images[2] = (images[2] * 0.2010 + 0.4465) * 255
images = images.transpose(1, 0, 2, 3)

# clip values not in [0,255]
images = np.clip(images, 0, 255)
print(np.min(images[0]), np.max(images[0]))
images = images.astype('uint8')
print(images.shape)

# change to B*H*W*C
images = images.transpose(0, 2, 3, 1)
print(images.shape)

# for i in range(images.shape[0]):
#     print(images[i])
#     img = Image.fromarray(images[i]).convert('RGB')
#     img.show()

print(res_count,dense_count)
np.save('./PGD_data.npy', images)
np.save('./PGD_label.npy', labels)
