# coding=gbk
"""
MyTransform��ʵ�ֶ����ݼ�Ԥ�����е��Զ���ת����������Ҫ�ı�__call__����������Ҫ�ı����е�myFunction����
myFunction����Ϊ����һ��ͼ�񣬶�ͼ�����Լ���Ҫ��ͼ����֮�󷵻ش���֮���ͼ������ͼ���������Ҫ����֪��
ʹ�÷���Ϊdataset = ImageFolder(root='./imagefolder', transform=MyTransform())

���ݼ����е�ת������ͨ��torchvision.transforms�еĸ���ת��������ʵ�֣������ת����������torchvision.functional��
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

# ʹ�÷���Ϊdataset = ImageFolder(root='./imagefolder', transform=MyTransform())
