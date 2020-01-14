# coding=gbk
from torch import nn
import torch
from data_process import Deflatten, Flatten
import matplotlib.pyplot as plt
"""--------------------------------------------生成器的结构如下--------------------------------------------
Fully connected layer from noise_dim to 1024 
ReLU 
Fully connected layer with size 1024 
ReLU 
Fully connected layer with size 784 
TanH（To clip the image to be [-1,1]）
---------------------------------------------最后产生的图像的像素在[-1,1]之间----------------------------------
"""


class Generator(nn.Module):

    def __init__(self, Noise_feature=100, out_feature=784):
        super(Generator, self).__init__()
        self.in_feature = Noise_feature
        self.out_feature = out_feature
        self.linear1 = nn.Linear(self.in_feature, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, self.out_feature)

    def forward(self, Noise):
        RES = torch.relu(self.linear1(Noise))
        RES = torch.relu(self.linear2(RES))
        RES = torch.tanh(self.linear3(RES))
        return RES


def generator(Noise_dim=100, out_feature=784):
    Model = nn.Sequential(
        nn.Linear(in_features=Noise_dim, out_features=1024),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=1024, out_features=1024),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=1024, out_features=out_feature),
        nn.Tanh(),
    )
    return Model


class DC_Generator(nn.Module):

    def __init__(self, Batch_size=128, Noise_feature=100):
        super(DC_Generator, self).__init__()
        self.batch_size = Batch_size
        self.fc1 = nn.Linear(in_features=Noise_feature, out_features=1024)
        self.relu1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(num_features=1)
        self.fc2 = nn.Linear(in_features=1024, out_features=7*7*128)
        # self.relu2 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm2d(num_features=1)
        self.reshape = Deflatten(self.batch_size, 128, 7, 7)
        self.conv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.batchnorm3 = nn.BatchNorm2d(num_features=64)
        self.conv2 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()
        self.flatten = Flatten()

    def forward(self, INPUT):
        # size = INPUT.size()
        RES = self.fc1(INPUT)
        RES = self.relu1(RES)
        RES = torch.reshape(RES, (-1, 1, RES.size()[0], RES.size()[1]))
        RES = self.batchnorm1(RES)
        RES = torch.reshape(RES, (RES.size()[2], RES.size()[3]))
        RES = self.fc2(RES)
        RES = torch.reshape(RES, (-1, 1, RES.size()[0], RES.size()[1]))
        RES = self.batchnorm2(RES)
        RES = torch.reshape(RES, (RES.size()[2], RES.size()[3]))
        RES = self.reshape(RES)
        RES = self.conv1(RES)
        RES = self.relu2(RES)
        RES = self.batchnorm3(RES)
        RES = self.conv2(RES)
        RES = self.tanh(RES)
        RES = self.flatten(RES)
        return RES


def build_dc_generator(noise_dim=96):
    """
    Build and return a PyTorch model implementing the DCGAN generator using
    the architecture described above.
    """
    return nn.Sequential(
        nn.Linear(noise_dim, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 7*7*128),
        nn.BatchNorm1d(7*7*128),
        Deflatten(128, 128, 7, 7),
        # deflatten(128, 128, 7, 7),
        nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=64),
        nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1),
        nn.Tanh(),
        Flatten(),
    )


# # model = generator(Noise_dim=10)  # 以函数的形式定义生成器
# model = Generator(10, 784)
# print(model)
# a = torch.randn(10)
# b = model(a).reshape((28, 28))
# # b = b.numpy()
# b = b.detach().numpy()
# plt.imshow(b, cmap='gray')
# plt.show()
# dcgan = DC_Generator(128, 100)
# inp = torch.randn(128, 100)
# outp = dcgan(inp)
# print(outp.size())
# inp2 = torch.randn(3, 100)
# model = DC_Generator(Batch_size=3, Noise_feature=100)
# output = model(inp2)
# print(output.size())
# gan = Generator(Noise_feature=100)
# output3 = gan(inp2)
# print(output3.size())
