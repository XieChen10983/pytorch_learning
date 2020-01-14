# coding=gbk
"""
�˴���ʵ���˶����������һ������ģ��Ĳ����ı䡣
������Ҫ���һ�������ģ�飬֮���ø�ֵ�ķ����ı䵥��Ĳ�����layer.weight.data = Tensor
��ȡ����ģ��ķ��������ȸ�������ÿ����ģ������ֻ�ȡ��ģ�飬֮�����ͨ�������õ���ģ�����ģ��==�𽥵õ�����
"""
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential(  # ��ģ��net
            nn.Sequential(  # ͨ��model.net[0]���Ի��
                nn.Linear(in_features=100, out_features=100),  # ͨ��model.net[0][0]���Ի��
                nn.Linear(in_features=100, out_features=100),
            ),
            nn.Linear(in_features=100, out_features=100),
            nn.Linear(in_features=100, out_features=100),
            nn.Linear(in_features=100, out_features=100),
            nn.Linear(in_features=100, out_features=100),
        )
        self.linear1 = nn.Linear(100, 200)  # ��ģ��linear1
        self.linear2 = nn.Linear(200, 100)  # ��ģ��linear2
        self.net2 = nn.Sequential(  # ��ģ��net2
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
