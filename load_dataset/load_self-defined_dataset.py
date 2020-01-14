# coding=gbk
"""
���е����ݼ�����torchvision.datasets.vision.VisionDataset�����࣬�����������ӷ�����
1. __len__(self)(��ȡ���ݼ��ĳ���):�������ݼ��ĳ���
2. __getitem__(self, index)(��ȡ���ݼ��е�ÿһ��):index����������ݼ��е������������������ݺͱ�ǩ��

���еķ���Ϊ��
DatasetFolder(root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None)
                 a. root����Ҫ��Ϊ���ݼ���·����root�µ��������ļ����µ�ͼ��Ϊһ�ࡣ
                 b. loaderΪһ������������һ��ͼ��·������һ��ͼ��
                 c. extensionsΪ��չ����ֻȡextensions�����е��ļ�
                 d. transform��target_transformΪĿ��任����
ImageFolder(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):ImageFolder��̳���DatasetFolder�ֻ࣬���ͼ������ݼ�
                 a. root����Ҫ��Ϊ���ݼ���·����root�¸��Ե����ļ����е�ͼ��Ϊһ��
                 b. loader��Ĭ�ϵ�ͼ���ȡ�����������øı�
                 c. transform��target_transformΪĿ��任����
"""
from torchvision.datasets import ImageFolder, DatasetFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets.vision import VisionDataset
import random
import numpy as np
import matplotlib.pyplot as plt
# from PIL import Image


def myFunction(img, probability, radius, center=None):
    img = np.array(img).astype(np.uint8)
    if center is None:
        # print()
        center = (round(img.shape[0]/2), round(img.shape[1]/2))
    if random.random() < probability:

        height, width = img.shape[:2]
        for i, j in zip(range(height), range(width)):
            if np.sqrt(np.square(i - center[0]) + np.square(j - center[1])) < radius:
                img[i, j, :] = 0
        # img = Image.fromarray(img.astype(np.uint8)).convert('RGB')

    return img


class MyTransform(object):
    def __init__(self, probability=0.5, radius=100, center=None):
        super(MyTransform, self).__init__()
        self.probability = probability
        self.radius = radius
        self.center = center

    def __call__(self, PIL_Image):
        # print(type(PIL_Image))
        return myFunction(PIL_Image, self.probability, self.radius, self.center)

    def __repr__(self):
        return self.__class__.__name__ + 'probability:{}, radius:{}'.format(self.probability, self.radius)


dataset = ImageFolder(root='./imagefolder', transform=MyTransform())
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
for image, label in dataloader:
    print(image.size(), label)
    plt.imshow(image[0])
    plt.title(str(label))
    plt.show()
