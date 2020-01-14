# coding=gbk
"""
此代码实现了对网络的任意一个单层模块的参数改变。
首先需要获得一个单层的模块，之后用赋值的方法改变单层的参数：layer.weight.data = Tensor
获取单层模块的方法：首先根据网络每个子模块的名字获取子模块，之后可以通过索引得到子模块的子模块==逐渐得到单层
"""
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential(  # 子模块net
            nn.Sequential(  # 通过model.net[0]可以获得
                nn.Linear(in_features=100, out_features=100),  # 通过model.net[0][0]可以获得
                nn.Linear(in_features=100, out_features=100),
            ),
            nn.Linear(in_features=100, out_features=100),
            nn.Linear(in_features=100, out_features=100),
            nn.Linear(in_features=100, out_features=100),
            nn.Linear(in_features=100, out_features=100),
        )
        self.linear1 = nn.Linear(100, 200)  # 子模块linear1
        self.linear2 = nn.Linear(200, 100)  # 子模块linear2
        self.net2 = nn.Sequential(  # 子模块net2
            nn.Linear(in_features=100, out_features=100),
            nn.Linear(in_features=100, out_features=100),
            nn.Linear(in_features=100, out_features=100),
            nn.Linear(in_features=100, out_features=100),
        )

    def forward(self, Input):
        result = self.net(Input)
        result = self.net2(self.linear2(self.linear1(result)))
        return result


import torch
model = Model()
print(model.net[1].weight.data.shape)
model.net[1].weight.data = torch.ones(size=(100, 100))
for parameter in model.net[1].weight.parameters():
    print(parameter)
# model2 = model
# print(model2.net[1].weight.data)
# print(model2.net[0][0])
