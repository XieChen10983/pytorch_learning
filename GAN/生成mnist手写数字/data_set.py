# coding=gbk
from torchvision import datasets as dset
from torch.utils.data import DataLoader
import torchvision.transforms as T


def mnist_dataloader(Batch_size=128):

    Mnist_train = dset.MNIST('B:/machine learning/deep learning/pytorch/GAN/生成mnist手写数字/cs231n/datasets/MNIST_data'
                             , train=True, download=False, transform=T.ToTensor())
    Train_loader = DataLoader(Mnist_train, batch_size=Batch_size)  # ,
    # sampler=ChunkSampler(NUM_TRAIN, 0)) # 从0位置开始采样NUM_TRAIN个数

    Mnist_val = dset.MNIST('B:/machine learning/deep learning/pytorch/GAN/生成mnist手写数字/cs231n/datasets/MNIST_data'
                           , train=True, download=False, transform=T.ToTensor())
    Val_loader = DataLoader(Mnist_val, batch_size=Batch_size)  # ,
    # sampler=ChunkSampler(NUM_VAL, NUM_TRAIN)) # 从NUM_TRAIN位置开始采样NUM_VAL个数
    return Train_loader, Val_loader
