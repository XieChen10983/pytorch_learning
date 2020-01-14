# coding=gbk
"""
���е����ݼ�����torch.utils.data.Dataset�����࣬�����������ӷ�����
1. __len__(��ȡ���ݼ��ĳ���)
2. __getItem__(��ȡ���ݼ��е�ÿһ��)

torchvision�е����ݼ�������
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

# ###################################################�ؼ����######################################################## #
dataset = CIFAR10(root=r'D:\�����ݴ��\data\initial_data\cifar', train=True,
                  transform=transform, target_transform=None, download=False)
# ###################################################�ؼ����######################################################## #

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
for image, label in dataloader:
    image = image[0]
    image = np.transpose(image, [1, 2, 0])
    plt.imshow(image, cmap='gray')
    plt.title(str(label))
    plt.show()
