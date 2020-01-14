# coding=gbk
"""
�˴�����ʾ����α�������ģ�ͼ����������λ�ȡ�����أ���������������ģ��
"""
import torch
from torch import nn


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


model = Model()  # ��ʼ����һ��model����
torch.save(obj=model, f='./model.pth')  # ������ģ�ͱ�����./model.pth�ϣ�ģ�Ͱ���ģ�͵Ľṹ�Ͳ�����

model2 = torch.load(f='./model.pth')  # �����������ģ�Ͷ����ص�model2�У�����������ͽṹ��

# ��print�������Կ���model��model2�Ľṹ��һ���ġ�
# ������֤model��model2�Ĳ���Ҳ��һ���ģ�����֤weight��bias����֤ͬ��ɵá�
model_weight = []
for layer in model.named_modules():  # model.named_modules()��������model�����е���ģ�鼰��ģ���е����ģ��
    # print(layer[0])
    if is_single_layer(layer[1]):  # ��������ֻ��Ҫ��֤ÿ�����ģ�飨�����㣩�Ĳ���һ�£�������Զ����is_single_layer�ж�
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
