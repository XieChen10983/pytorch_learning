# coding=gbk
"""
此代码演示了如何在已有的网络的基础上部分改动网络以及参数
"""
from torchvision.models import vgg11
import torch.nn as nn

# vgg11是torchvision中事先实现的网络，网络的结构包括三个子模块，用vgg.features、vgg.avgpool、vgg.classifier可以分别得到这些模块
vgg = vgg11(pretrained=False)
# print(vgg)


# 在vgg模块的基础上改造自己的网络
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.get_feature = vgg.features  # 将vgg的features模块作为自己的get_feature模块（包含了参数）
        self.classifier = nn.Linear(in_features=512, out_features=1)  # 添加自己的模块

    def forward(self, Input):
        batch_size = Input.size(0)
        result = self.get_feature(Input)
        result = self.classifier(result.view(size=(batch_size, -1)))
        return result


import torch
model = Model()
print(model)  # 打印模型可以发现模型结构中的get_feature和vgg中的features模块是一样的
input = torch.randn(size=(5, 3, 32, 32))
output = model(input)
print(output.shape)
