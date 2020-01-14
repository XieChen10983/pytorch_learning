# coding=gbk
"""
�˴�����ʾ����������е�����Ļ����ϲ��ָĶ������Լ�����
"""
from torchvision.models import vgg11
import torch.nn as nn

# vgg11��torchvision������ʵ�ֵ����磬����Ľṹ����������ģ�飬��vgg.features��vgg.avgpool��vgg.classifier���Էֱ�õ���Щģ��
vgg = vgg11(pretrained=False)
# print(vgg)


# ��vggģ��Ļ����ϸ����Լ�������
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.get_feature = vgg.features  # ��vgg��featuresģ����Ϊ�Լ���get_featureģ�飨�����˲�����
        self.classifier = nn.Linear(in_features=512, out_features=1)  # ����Լ���ģ��

    def forward(self, Input):
        batch_size = Input.size(0)
        result = self.get_feature(Input)
        result = self.classifier(result.view(size=(batch_size, -1)))
        return result


import torch
model = Model()
print(model)  # ��ӡģ�Ϳ��Է���ģ�ͽṹ�е�get_feature��vgg�е�featuresģ����һ����
input = torch.randn(size=(5, 3, 32, 32))
output = model(input)
print(output.shape)
