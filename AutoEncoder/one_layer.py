# coding=gbk
from data_set import mnist_dataloader
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, gpu_status):
        super(Net, self).__init__()
        self.gpu_status = gpu_status
        self.hidden = 10

        self.en_conv_1 = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.Tanh()
        )
        self.en_conv_2 = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.Tanh()
        )

        self.en_fc_1 = nn.Linear(16*7*7, self.hidden)
        self.en_fc_2 = nn.Linear(16*7*7, self.hidden)

        self.de_fc = nn.Linear(self.hidden, 16*7*7)
        self.de_conv = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def encoder(self, INPUT):
        conv_out_1 = self.en_conv_1(INPUT)
        conv_out_1 = conv_out_1.view(INPUT.size(0), -1)

        conv_out_2 = self.en_conv_2(INPUT)
        conv_out_2 = conv_out_2.view(INPUT.size(0), -1)

        encoded_fc1 = self.en_fc_1(conv_out_1)
        encoded_fc2 = self.en_fc_1(conv_out_2)

        return encoded_fc1, encoded_fc2  # 这里分别表示均值和标准差的采样

    def sampler(self, MEAN, STD):
        Var = STD.mul(0.5).exp_()
        Eps = torch.FloatTensor(Var.size()).normal_()
        if self.gpu_status:
            Eps = Eps.cuda()
        return Eps.mul(Var).add_(MEAN)

    def decoder(self, INPUT):
        OUT = self.de_fc(INPUT)
        OUT = OUT.view(-1, 16, 7, 7)
        OUT = self.de_conv(OUT)
        return OUT

    def forward(self, INPUT):
        mean, std = self.encoder(INPUT)
        code = self.sampler(mean, std)
        out = self.decoder(code)
        return out, code, mean, std


net = Net(1)
print(net)
