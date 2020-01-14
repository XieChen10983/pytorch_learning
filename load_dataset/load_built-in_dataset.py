# coding=gbk
"""
所有的数据集都是torch.utils.data.Dataset的子类，都包含以下子方法：
1. __len__(获取数据集的长度)
2. __getItem__(获取数据集中的每一项)

torchvision中的数据集包括：
'LSUN', 'LSUNClass',
           'ImageFolder', 'DatasetFolder', 'FakeData',
           'CocoCaptions', 'CocoDetection',
           'CIFAR10', 'CIFAR100', 'EMNIST', 'FashionMNIST', 'QMNIST',
           'MNIST', 'KMNIST', 'STL10', 'SVHN', 'PhotoTour', 'SEMEION',
           'Omniglot', 'SBU', 'Flickr8k', 'Flickr30k',
           'VOCSegmentation', 'VOCDetection', 'Cityscapes', 'ImageNet',
           'Caltech101', 'Caltech256', 'CelebA', 'SBDataset', 'VisionDataset',
           'USPS', 'Kinetics400', 'HMDB51', 'UCF101'
"""
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

plt.figure()
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(0.7),
    transforms.RandomVerticalFlip(0.7),
    transforms.ToTensor(),
])

# ###################################################关键语句######################################################## #
dataset = CIFAR10(root=r'D:\大数据存放\data\initial_data\cifar', train=True,
                  transform=transform, target_transform=None, download=False)
# ###################################################关键语句######################################################## #

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
for image, label in dataloader:
    image = image[0]
    image = np.transpose(image, [1, 2, 0])
    plt.imshow(image, cmap='gray')
    plt.title(str(label))
    plt.show()
