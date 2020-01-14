# coding=gbk
from combine2gan import gan
from discriminator import DC_Discriminator
from generator import DC_Generator
from get_optimizer import get_optimizer
from discriminator_loss import discriminator_loss
from generator_loss import generator_loss
import matplotlib.pyplot as plt
import numpy as np
from data_set import mnist_dataloader
# from gen_random_noise import gen_noise

# batch_size = 128
# mnist_train = dset.MNIST('./cs231n/datasets/MNIST_data', train=True, download=False, transform=T.ToTensor())
# train_loader = DataLoader(mnist_train, batch_size=batch_size)  # ,
# # sampler=ChunkSampler(NUM_TRAIN, 0)) # 从0位置开始采样NUM_TRAIN个数
#
# mnist_val = dset.MNIST('./cs231n/datasets/MNIST_data', train=True, download=False, transform=T.ToTensor())
# val_loader = DataLoader(mnist_val, batch_size=batch_size)  # ,
# # sampler=ChunkSampler(NUM_VAL, NUM_TRAIN)) # 从NUM_TRAIN位置开始采样NUM_VAL个数
train_loader, _ = mnist_dataloader()

D = DC_Discriminator(Batch_size=128)
G = DC_Generator(Batch_size=128, Noise_feature=100)
D_optimizer = get_optimizer(D)
G_optimizer = get_optimizer(G)

epoch_image = gan(train_loader, D, G, D_optimizer, G_optimizer, discriminator_loss, generator_loss,
                  Noise_feature=100, Num_epochs=10)

plt.figure(1, figsize=(10, 10))
num = len(epoch_image)
for index, image in enumerate(epoch_image):
    plt.subplot(np.ceil(np.sqrt(num)), np.ceil(np.sqrt(num)), index+1)
    plt.imshow(image, cmap='gray')
plt.show()
