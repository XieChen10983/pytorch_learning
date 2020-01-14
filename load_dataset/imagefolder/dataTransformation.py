# coding=gbk
"""
MyTransform类实现对数据集预处理中的自定义转换，其中需要改变__call__函数，即需要改变其中的myFunction函数
myFunction函数为输入一张图像，对图像做自己需要的图像处理之后返回处理之后的图像，输入图像的类型需要事先知道
使用方法为dataset = ImageFolder(root='./imagefolder', transform=MyTransform())

数据集已有的转换可以通过torchvision.transforms中的各个转换的类来实现，其各个转换函数则在torchvision.functional中
"""
import numpy as np
import random


def myFunction(img, probability, radius, center=None):
    img = np.array(img).astype(np.uint8)
    if center is None:
        # print()
        center = (round(img.shape[0]/2), round(img.shape[1]/2))
    if random.random() < probability:

        height, width = img.shape[:2]
        # area = np.where(img-)
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

# 使用方法为dataset = ImageFolder(root='./imagefolder', transform=MyTransform())
