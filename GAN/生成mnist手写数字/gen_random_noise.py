# coding=gbk
import torch
from generator import Generator


def gen_noise(Batch_size, Noise_feature):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, dim) containing uniform
      random noise in the range (-1, 1).
    """
    Noise_for_generator = torch.rand(Batch_size, Noise_feature) + torch.rand(Batch_size, Noise_feature)*(-1)

    return Noise_for_generator
#
#
# noise = gen_noise(Batch_size=32, Noise_feature=10)
# model = Generator(Noise_feature=10, out_feature=784)
# print(noise)
# print(noise.size())
# output = model(noise)
# print(output.size())
