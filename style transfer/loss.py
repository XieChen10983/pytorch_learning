# coding=gbk
from torch import nn
import torch


class Content_Loss(nn.Module):
    def __init__(self, target, weight):
        super(Content_Loss, self).__init__()
        self.weight = weight
        self.target = target.detach() * self.weight
        # 必须要用detach来分离出target，这时候target不再是一个Variable，可以动态计算梯度，否则forward会出错，不能向前传播
        self.criterion = nn.MSELoss()
        self.loss = None

    def forward(self, INPUT):
        self.loss = self.criterion(INPUT * self.weight, self.target)
        out = INPUT.clone()
        return out

    def backward(self, retain_variables=True):
        self.loss.backward(retain_variables=retain_variables)
        return self.loss


class Gram(nn.Module):
    def __init__(self):
        super(Gram, self).__init__()

    def forward(self, INPUT):
        a, b, c, d = INPUT.size()
        feature = INPUT.view(a*b, c*d)
        gram = torch.mm(feature, feature.t())
        gram /= (a*b*c*d)
        return gram


class Style_Loss(nn.Module):
    def __init__(self, target, weight):
        super(Style_Loss, self).__init__()
        self.weight = weight
        self.target = target
        self.gram = Gram()
        self.loss = None
        self.criterion = nn.MSELoss()

    def forward(self, INPUT):
        G = self.gram(INPUT) * self.weight
        self.loss = self.criterion(G, self.target)
        out = INPUT.clone()
        return out

    def backward(self, retain_variables=True):
        self.loss.backward(retain_variables=retain_variables)
        return self.loss
