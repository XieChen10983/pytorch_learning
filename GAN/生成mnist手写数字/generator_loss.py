# coding=gbk
import torch
from torch.nn import BCEWithLogitsLoss
import numpy as np


def generator_loss(Fake_logits, loss_function=BCEWithLogitsLoss()):
    # Batch size
    Logits_num = Fake_logits.size()

    # 生成器的作用是将所有的假的向真的靠拢，真的的标签为全1
    True_logits = torch.ones(Logits_num).float()

    # 计算生成器的损失
    Loss = loss_function(Fake_logits, True_logits)
    return Loss


# a = torch.randn(10)
# b = generator_loss(a)
# print(b)
# a = np.random.random((10, 1))
# for index, data in enumerate(a):
#     print(index)
#     print(data)
