# coding=gbk
from torch import nn
from data_process import Flatten, Deflatten
import torch
"""---------------------------------------------判别器的结构如下-----------------------------------------------
Fully connected layer from size 784 to 256 
LeakyReLU with alpha 0.01 
Fully connected layer from 256 to 256 
LeakyReLU with alpha 0.01 
Fully connected layer from 256 to 1
"""


class Discriminator(nn.Module):  # 以类的形式生成鉴别器

    def __init__(self, Input_dim=784):
        super(Discriminator, self).__init__()
        self.in_feature = Input_dim
        self.out_feature = 1
        self.flatten = Flatten()
        self.linear1 = nn.Linear(self.in_feature, 256)
        self.leakyrelu1 = nn.LeakyReLU(0.01, inplace=True)
        self.linear2 = nn.Linear(256, 256)
        self.leakyrelu2 = nn.LeakyReLU(0.01, inplace=True)
        self.linear3 = nn.Linear(256, self.out_feature)

    def forward(self, Tensor):
        print('using gan')
        RES = self.flatten(Tensor)
        RES = self.linear1(RES)
        RES = self.leakyrelu1(RES)
        RES = self.linear2(RES)
        RES = self.leakyrelu2(RES)
        Score = self.linear3(RES)
        return Score


def discriminator(Image):   # 以函数的形式生成鉴别器
    Model = nn.Sequential(
        Flatten(),
        nn.Linear(in_features=Flatten()(Image).size()[1], out_features=256),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Linear(in_features=256, out_features=256),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Linear(in_features=256, out_features=1),
    )
    return Model


class DC_Discriminator(nn.Module):
    def __init__(self, Batch_size=128, Input_dim=784):
        super(DC_Discriminator, self).__init__()
        self.batch_size = Batch_size
        self.input_dim = Input_dim
        self.deflatten = Deflatten(self.batch_size, 1, 28, 28)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)
        self.leakyrelu1 = nn.LeakyReLU(negative_slope=0.01)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.leakyrelu2 = nn.LeakyReLU(negative_slope=0.01)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()
        self.fc1 = nn.Linear(in_features=4*4*64, out_features=4*4*64)
        self.leakyrelu3 = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(in_features=4*4*64, out_features=1)

    def forward(self, INPUT):
        print('using dc')
        RES = self.deflatten(INPUT)
        RES = self.conv1(RES)
        RES = self.leakyrelu1(RES)
        RES = self.maxpool1(RES)
        RES = self.conv2(RES)
        RES = self.leakyrelu2(RES)
        RES = self.maxpool2(RES)
        RES = self.flatten(RES)
        RES = self.fc1(RES)
        RES = self.leakyrelu3(RES)
        RES = self.fc2(RES)
        return RES


#
# a = torch.randn((32, 1, 28, 28))
# d = Discriminator(28*28)
# # for i in range(1000):
# #     a = torch.randn((1, 1, 28, 28))
# #     if (d(a) > 1) or (d(a) < 0):
# #         print(d(a))
# c = d(a)
# print(c)
# print(c.size())
# print('wancheng')
inp = torch.randn(128, 1, 28, 28)
dcgan = DC_Discriminator()
output = dcgan(inp)
print(output.size())
