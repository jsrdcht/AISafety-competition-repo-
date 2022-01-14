import numpy as np
import torchvision
import random
import sys
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
images = []
soft_labels = []
for image, label in dataset:
    image = np.array(image)
    images.append(image)
    soft_label = np.zeros(10)
    soft_label[label] += random.uniform(0, 10)  # an unnormalized soft label vector
    soft_labels.append(soft_label)

    # if len(soft_labels)<10: print(label)
images = np.array(images)
soft_labels = np.array(soft_labels)


dataset2 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
images2 = []
soft_labels2 = []
for image, label in dataset2:
    image = np.array(image)
    images2.append(image)
    soft_label = np.zeros(10)
    soft_label[label] += random.uniform(0, 10)  # an unnormalized soft label vector
    soft_labels2.append(soft_label)

    # if len(soft_labels)<10: print(label)
images2 = np.array(images2)
soft_labels2 = np.array(soft_labels2)

images=np.concatenate([images,images2])
soft_labels=np.concatenate([soft_labels,soft_labels2])

train_images=images[10000:]
train_labels=soft_labels[10000:]

test_images=images[:10000]
test_labels=soft_labels[:10000]

print(sys.getsizeof(train_images))
print(test_images.shape, images.dtype, test_labels.shape, soft_labels.dtype)
# np.save('cifar_image_train.npy', train_images)
# np.save('cifar_label_train.npy', train_labels)
# np.save('cifar_image_test.npy', test_images)
# np.save('cifar_label_test.npy', test_labels)
