import torch
import numpy as np
import torch as t
from torch import nn
from torch.autograd import Variable as V
a = torch.randn(size=(3, 3, 32, 32))
print(a[0].size())
b = a.resize_(a[0].size()).copy_(a[0])
print(b.size())
