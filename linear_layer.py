# coding=gbk
from torch.nn import Linear, CrossEntropyLoss
import torch
import torch.nn.functional as F
import numpy as np

# myLayer1 = Linear(in_features=10, out_features=5, bias=True)
# myLayer2 = Linear(5, 2)
# in_fea = torch.randn(10)
# out_fea = myLayer2(myLayer1(in_fea))
# print(out_fea)
# print('---------------------------------------weight--------------------------------------')
# print(myLayer.weight)
# print('---------------------------------------bias--------------------------------------')
# print(myLayer.bias)
# print('---------------------------------------parameters--------------------------------------')
# print(myLayer.parameters())

loss_func = CrossEntropyLoss()
tensor = torch.randn((32, 10), requires_grad=True)
target = torch.randn((32, 2), requires_grad=True)
print(tensor.size())
print(target.size())
# loss = F.nll_loss(tensor, target)
# loss = loss_func(input=tensor, target=target)
# print(loss)
print(np.arange(1, 10+1).shape)
