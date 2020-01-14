# coding=gbk
import torch
from torch.nn import BCEWithLogitsLoss
import numpy as np


def generator_loss(Fake_logits, loss_function=BCEWithLogitsLoss()):
    # Batch size
    Logits_num = Fake_logits.size()

    # �������������ǽ����еļٵ�����Ŀ�£����ĵı�ǩΪȫ1
    True_logits = torch.ones(Logits_num).float()

    # ��������������ʧ
    Loss = loss_function(Fake_logits, True_logits)
    return Loss


# a = torch.randn(10)
# b = generator_loss(a)
# print(b)
# a = np.random.random((10, 1))
# for index, data in enumerate(a):
#     print(index)
#     print(data)
