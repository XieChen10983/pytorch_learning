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

transform = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
dataset = ImageFolder(root='./imagefolder', transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
for image, label in dataloader:
    print(image.size(), label)
