# coding=gbk
from torch import nn
import torch


class Flatten(nn.Module):
    def forward(self, Tensor):
        Batch, _, _, _ = Tensor.size()
        return Tensor.view(Batch, -1)  # ÿһ���е��������ֲ��䣬ͼ��չƽ��һ������


def flatten(Tensor):
    Batch, _, _, _ = Tensor.size()
    return Tensor.view(Batch, -1)  # ÿһ���е��������ֲ��䣬ͼ��չƽ��һ������


class Deflatten(nn.Module):
    """
    An Deflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """
    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Deflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)


def deflatten(Tensor, Channel=128, Height=7, Width=7):
    return Tensor.view(-1, Channel, Height, Width)  # ÿһ���е��������ֲ��䣬ͼ��չƽ��һ������


# deflatten1 = Deflatten()
# a = torch.randn((32, 128*7*7))
# b = deflatten(a)
# c = deflatten(a)
# print(a.size())
# print(b.size())
# print(c.size())
