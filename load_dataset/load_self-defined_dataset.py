# coding=gbk
"""
所有的数据集都是torchvision.datasets.vision.VisionDataset的子类，都包含以下子方法：
1. __len__(self)(获取数据集的长度):返回数据集的长度
2. __getitem__(self, index)(获取数据集中的每一项):index必须遍历数据集中的所有样本，返回数据和标签。

已有的方法为：
DatasetFolder(root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None)
                 a. root是想要作为数据集的路径，root下的所有子文件夹下的图像为一类。
                 b. loader为一个函数，输入一个图像路径返回一个图像
                 c. extensions为扩展名，只取extensions里面有的文件
                 d. transform、target_transform为目标变换函数
ImageFolder(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):ImageFolder类继承于DatasetFolder类，只针对图像的数据集
                 a. root是想要作为数据集的路径，root下各自的子文件夹中的图像为一类
                 b. loader有默认的图像读取器，基本不用改变
                 c. transform和target_transform为目标变换函数
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
