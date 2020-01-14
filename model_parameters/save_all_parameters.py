# coding=gbk
"""
�˴�����ʾ����α�������ģ�͵Ĳ�������������ṹ�����������ģ�͵Ĳ�������Ҫ�ȹ������Ӧ�Ľṹ��
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
    �˺����ж�ģ��Module�Ƿ�Ϊ����ģ�顣
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
    ��Ϊ�Զ��������һ��ģ��
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
torch.save(model1.state_dict(), './state_dict.pth')  # ����ģ�͵Ĳ�����������ģ�͵Ľṹ���Ժ���Ҫ�õ��ò���ʱ��Ҫ�ȹ�����Ӧ�ṹ

model2 = Model()
model3 = Model()
# model3����model1������Ĳ�����ע���÷�����torch.load()���غ���Ӧ�Ĳ����������model.load_state_dict()���ز���
model3.load_state_dict(torch.load(f='./state_dict.pth'))

# ������֤�Ѿ�������model1�Ĳ���������֤Ȩ��weight��ͬ��
model_weight = [[], [], []]
model = [model1, model2, model3]
for i, sub_model in enumerate(model):
    for layer in sub_model.named_modules():
        # print(layer[0])
        if is_single_layer(layer[1]):  # �ж��Ƿ�Ϊ����
            if not ((isinstance(layer[1], nn.ReLU)) or (isinstance(layer[1], nn.MaxPool2d))):  # �ж��Ƿ���Ȩ��
                # print(layer[1].weight.shape)
                model_weight[i].append(layer[1].weight.data)

print(model_weight[0][0] == model_weight[1][0])  # ��������Ӧ����ȫFalse
print('='*30)
print(model_weight[0][0] == model_weight[2][0])  # ��������Ӧ����ȫTrue
