# coding=gbk
"""
此代码演示了如何保存整个模型的参数（不包括其结构）、如何下载模型的参数（需要先构造好相应的结构）
"""
from torch import nn
import torch


def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param mod:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)


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


model1 = Model()
torch.save(model1.state_dict(), './state_dict.pth')  # 保存模型的参数，不包括模型的结构，以后需要用到该参数时需要先构造相应结构

model2 = Model()
model3 = Model()
# model3采用model1所保存的参数，注意用法，先torch.load()下载好相应的参数，最后用model.load_state_dict()加载参数
model3.load_state_dict(torch.load(f='./state_dict.pth'))

# 下面验证已经加载了model1的参数（仅验证权重weight相同）
model_weight = [[], [], []]
model = [model1, model2, model3]
for i, sub_model in enumerate(model):
    for layer in sub_model.named_modules():
        # print(layer[0])
        if is_single_layer(layer[1]):  # 判断是否为单层
            if not ((isinstance(layer[1], nn.ReLU)) or (isinstance(layer[1], nn.MaxPool2d))):  # 判断是否有权重
                # print(layer[1].weight.shape)
                model_weight[i].append(layer[1].weight.data)

print(model_weight[0][0] == model_weight[1][0])  # 不出意外应该是全False
print('='*30)
print(model_weight[0][0] == model_weight[2][0])  # 不出意外应该是全True
