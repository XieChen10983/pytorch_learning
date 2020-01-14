# coding=gbk
"""
此代码实现整个网络的参数初始化。
首先定义初始化函数weights_init
用model.apply(weight_init)的方法，将逐层应用weight_init实现初始化
"""
from torch import nn


def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param mod:
    :return:
    """
    classname = mod.__class__.__name__  # .__class__.__name__的方法可以得到单层模块的名称
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)


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


model = Model()
model.apply(weights_init)
print(model)


module = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
classname = module.__class__.__name__
print(classname.find('Conv2d'))
