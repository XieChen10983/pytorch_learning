# coding=gbk
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset

# import numpy as np
#
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
#
# plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'
#
#
# def show_images(images):
#     images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
#     sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
#     sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))
#
#     plt.figure(figsize=(sqrtn, sqrtn))
#     gs = gridspec.GridSpec(sqrtn, sqrtn)
#     gs.update(wspace=0.05, hspace=0.05)
#
#     for i, img in enumerate(images):
#         ax = plt.subplot(gs[i])
#         plt.axis('off')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_aspect('equal')
#         plt.imshow(img.reshape([sqrtimg, sqrtimg]))
#     return
#
#
# def preprocess_img(x):
#     return 2 * x - 1.0
#
#
# def deprocess_img(x):
#     return (x + 1.0) / 2.0
#
#
# def rel_error(x, y):
#     return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
#
#
# def count_params(model):
#     """Count the number of parameters in the current TensorFlow graph """
#     param_count = np.sum([np.prod(p.size()) for p in model.parameters()])
#     return param_count
#
#
# # 采样函数为自己定义的序列采样（即按顺序采样）
# class ChunkSampler(sampler.Sampler):
#     """Samples elements sequentially from some offset.
#     Arguments:
#         num_samples: # of desired datapoints
#         start: offset where we should start selecting from
#     """
#     def __init__(self, num_samples, start=0):
#         self.num_samples = num_samples
#         self.start = start
#
#     def __iter__(self):
#         return iter(range(self.start, self.start + self.num_samples))
#
#     def __len__(self):
#         return self.num_samples
#
#
# NUM_TRAIN = 50000   # 训练集数量
# NUM_VAL = 5000      # 测试集数量
#
# NOISE_DIM = 96
# batch_size = 128
#
# mnist_train = dset.MNIST('./cs231n/datasets/MNIST_data', train=True, download=True, transform=T.ToTensor())
# loader_train = DataLoader(mnist_train, batch_size=batch_size,
#                           sampler=ChunkSampler(NUM_TRAIN, 0))  # 从0位置开始采样NUM_TRAIN个数
#
# mnist_val = dset.MNIST('./cs231n/datasets/MNIST_data', train=True, download=True, transform=T.ToTensor())
# loader_val = DataLoader(mnist_val, batch_size=batch_size,
#                         sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))  # 从NUM_TRAIN位置开始采样NUM_VAL个数
#
#
# # imgs = loader_train.__iter__().next()[0].view(batch_size, 784).numpy().squeeze()
# # show_images(imgs)
# def sample_noise(Batch_size, dim):
#     """
#     Generate a PyTorch Tensor of uniform random noise.
#
#     Input:
#     - batch_size: Integer giving the batch size of noise to generate.
#     - dim: Integer giving the dimension of noise to generate.
#
#     Output:
#     - A PyTorch Tensor of shape (batch_size, dim) containing uniform
#       random noise in the range (-1, 1).
#     """
#     temp = torch.rand(Batch_size, dim) + torch.rand(Batch_size, dim)*(-1)
#
#     return temp
#
#
# class Flatten(nn.Module):
#     def forward(self, x):
#         N, C, H, W = x.size()  # read in N, C, H, W
#         return x.view(N, -1)   # "flatten" the C * H * W values into a single vector per image
#
#
# class Unflatten(nn.Module):
#     """
#     An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
#     to produce an output of shape (N, C, H, W).
#     """
#     def __init__(self, N=-1, C=128, H=7, W=7):
#         super(Unflatten, self).__init__()
#         self.N = N
#         self.C = C
#         self.H = H
#         self.W = W
#
#     def forward(self, x):
#         return x.view(self.N, self.C, self.H, self.W)
#
#
# def initialize_weights(m):
#     if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
#         init.xavier_uniform(m.weight)
#
#
# def discriminator():
#     """
#     Build and return a PyTorch model implementing the architecture above.
#     """
#     model = nn.Sequential(
#         Flatten(),
#         nn.Linear(784, 256),
#         nn.LeakyReLU(0.01, inplace=True),
#         nn.Linear(256, 256),
#         nn.LeakyReLU(0.01, inplace=True),
#         nn.Linear(256, 1)
#     )
#     return model
#
#
# def generator(noise_dim=NOISE_DIM):
#     """
#     Build and return a PyTorch model implementing the architecture above.
#     """
#     model = nn.Sequential(
#         nn.Linear(noise_dim, 1024),
#         nn.ReLU(inplace=True),
#         nn.Linear(1024, 1024),
#         nn.ReLU(inplace=True),
#         nn.Linear(1024, 784),
#         nn.Tanh(),
#     )
#     return model
#
#
# def discriminator_loss(logits_real, logits_fake):
#     """
#     Computes the discriminator loss described above.
#
#     Inputs:
#     - logits_real: PyTorch Variable of shape (N,) giving scores for the real data.
#     - logits_fake: PyTorch Variable of shape (N,) giving scores for the fake data.
#
#     Returns:
#     - loss: PyTorch Variable containing (scalar) the loss for the discriminator.
#     """
#     # loss = None
#     bce_loss = nn.BCEWithLogitsLoss()
#     # Batch size.
#     N = logits_real.size()
#
#     # 目标label，全部设置为1意味着判别器需要做到的是将正确的全识别为正确，错误的全识别为错误
#     # true_labels = Variable(torch.ones(N)).long()
#     true_labels = torch.ones(N).long()
#
#     real_image_loss = bce_loss(logits_real.float(), true_labels.float())  # 识别正确的为正确
#     fake_image_loss = bce_loss(logits_fake, torch.ones(N).long() - true_labels)  # 识别错误的为错误
#
#     loss = real_image_loss + fake_image_loss
#
#     return loss
#
#
# def generator_loss(logits_fake):
#     """
#     Computes the generator loss described above.
#
#     Inputs:
#     - logits_fake: PyTorch Variable of shape (N,) giving scores for the fake data.
#
#     Returns:
#     - loss: PyTorch Variable containing the (scalar) loss for the generator.
#     """
#     bce_loss = nn.BCEWithLogitsLoss()
#     # Batch size.
#     N = logits_fake.size()
#
#     # 生成器的作用是将所有“假”的向真的（1）靠拢
#     # true_labels = Variable(torch.ones(N)).long()
#     true_labels = torch.ones(N).long()
#
#     # 计算生成器损失
#     loss = bce_loss(logits_fake, true_labels)
#
#     return loss
#
#
# def get_optimizer(model):
#     """
#     Construct and return an Adam optimizer for the model with learning rate 1e-3,
#     beta1=0.5, and beta2=0.999.
#
#     Input:
#     - model: A PyTorch model that we want to optimize.
#
#     Returns:
#     - An Adam optimizer for the model with the desired hyperparameters.
#     """
#     optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
#     return optimizer
#
#
# def run_a_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss, show_every=250,
#               batch_size=128, noise_size=96, num_epochs=10):
#     """
#     Train a GAN!
#
#     Inputs:
#     - D, G: pytorch模块，分别为判别器和生成器
#     - D_solver, G_solver: torch.optim Optimizers to use for training the
#       discriminator and generator.
#     - discriminator_loss, generator_loss: Functions to use for computing the generator and
#       discriminator loss, respectively.
#     - show_every: Show samples after every show_every iterations.
#     - batch_size: Batch size to use for training.
#     - noise_size: Dimension of the noise to use as input to the generator.
#     - num_epochs: Number of epochs over the training dataset to use for training.
#     """
#     iter_count = 0
#     for epoch in range(num_epochs):
#         for x, _ in loader_train:
#             if len(x) != batch_size:
#                 continue
#             D_solver.zero_grad()
#             # real_data = Variable(x).type(dtype)
#             real_data = x.long()
#             # logits_real = D(2* (real_data - 0.5)).type(dtype)
#             logits_real = D((2 * (real_data - 0.5)).float())
#
#             # g_fake_seed = Variable(sample_noise(batch_size, noise_size)).type(dtype)
#             g_fake_seed = sample_noise(batch_size, noise_size).long()
#             fake_images = G(g_fake_seed.float()).detach()
#             logits_fake = D(fake_images.view(batch_size, 1, 28, 28))
#
#             print(logits_real.float(), logits_fake.float())
#             d_total_error = discriminator_loss(logits_real.float(), logits_fake.float())
#             d_total_error.backward()
#             D_solver.step()
#
#             G_solver.zero_grad()
#             # g_fake_seed = Variable(sample_noise(batch_size, noise_size)).type(dtype)
#             g_fake_seed = sample_noise(batch_size, noise_size).long()
#             fake_images = G(g_fake_seed)
#
#             gen_logits_fake = D(fake_images.view(batch_size, 1, 28, 28))
#             g_error = generator_loss(gen_logits_fake)
#             g_error.backward()
#             G_solver.step()
#
#             if iter_count % show_every == 0:
#                 print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_total_error.data[0], g_error.data[0]))
#                 imgs_numpy = fake_images.data.cpu().numpy()
#                 show_images(imgs_numpy[0:16])
#                 plt.show()
#                 print()
#             iter_count += 1
#
#
# # Make the discriminator
# # D = discriminator().type(dtype)
# D = discriminator()
#
# # Make the generator
# # G = generator().type(dtype)
# G = generator()
#
# # Use the function you wrote earlier to get optimizers for the Discriminator and the Generator
# D_solver = get_optimizer(D)
# G_solver = get_optimizer(G)
# # Run it!
# run_a_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss)
a = torch.randn(size=(2, 10))
print(a)
b = torch.relu(a)
c = torch.relu_(a)
print(b)
print(c)
