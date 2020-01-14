# coding=gbk
"""
此代码演示了如何保存整个模型及其参数、如何获取（下载）整个包含参数的模型
"""
import torch
from torch import nn


def is_single_layer(Module):
    """
    此函数判断模块Module是否为单层模块。
    :param Module:
    :return:
    """
    layers = list(Module.named_modules())
    if len(layers) == 1:
        return True
    else:
        return False


class Model(nn.Module):
    """
    此为自定义的任意一个模型
    """
    def __init__(self):
        super(Model, self).__init__()
        self.feature1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
        self.feature2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3),
            nn.BatchNorm2d(5),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3),
            nn.BatchNorm2d(5),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.MaxPool2d(2, 2),
        )

    def forward(self, Input):
        output = self.classifier(self.feature2(self.feature1(Input)))
        return output


model = Model()  # 初始化了一个model对象
torch.save(obj=model, f='./model.pth')  # 将整个模型保存在./model.pth上，模型包括模型的结构和参数。

model2 = torch.load(f='./model.pth')  # 将整个保存的模型都下载到model2中，包括其参数和结构。

# 用print函数可以看出model和model2的结构是一样的。
# 下面验证model和model2的参数也是一样的：仅验证weight，bias的验证同理可得。
model_weight = []
for layer in model.named_modules():  # model.named_modules()函数返回model中所有的子模块及子模块中的深层模块
    # print(layer[0])
    if is_single_layer(layer[1]):  # 现在我们只需要验证每个深层模块（即单层）的参数一致，因此用自定义的is_single_layer判断
        if not ((isinstance(layer[1], nn.ReLU)) or (isinstance(layer[1], nn.MaxPool2d))):
            # print(layer[1].weight.shape)
            model_weight.append(layer[1].weight.data)

model2_weight = []
for layer in model2.named_modules():
    # print(layer[0])
    if is_single_layer(layer[1]):
        if not ((isinstance(layer[1], nn.ReLU)) or (isinstance(layer[1], nn.MaxPool2d))):
            # print(layer[1].weight.shape)
            model2_weight.append(layer[1].weight.data)

print(model_weight[2] == model2_weight[2])
# print(torch.all(model2_weight == model_weight))
