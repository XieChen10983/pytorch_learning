# coding=gbk
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.generate = nn.Linear(1, 10)

    def forward(self, Input):
        result = self.generate(Input)
        return result


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminate = nn.Linear(10, 1)

    def forward(self, Input):
        result = self.discriminate(Input)
        return result


def freeze_model(Model):
    for param in Model.parameters():
        param.required_grad = False


def unfreeze_model(Model):
    for param in Model.parameters():
        param.required_grad = True


generator = Generator()
discriminator = Discriminator()

input1 = torch.randn(size=(5, 1))
for i in range(10):
    freeze_model(discriminator)
    gen = generator(input1)
    unfreeze_model(discriminator)
    freeze_model(generator)
    dis = discriminator(gen)
