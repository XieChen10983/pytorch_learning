# coding=gbk
from torchvision import datasets as dset
from torch.utils.data import DataLoader
import torchvision.transforms as T


def mnist_dataloader(Batch_size=128):

    Mnist_train = dset.MNIST('B:/machine learning/deep learning/pytorch/GAN/����mnist��д����/cs231n/datasets/MNIST_data'
                             , train=True, download=False, transform=T.ToTensor())
    Train_loader = DataLoader(Mnist_train, batch_size=Batch_size)  # ,
    # sampler=ChunkSampler(NUM_TRAIN, 0)) # ��0λ�ÿ�ʼ����NUM_TRAIN����

    Mnist_val = dset.MNIST('B:/machine learning/deep learning/pytorch/GAN/����mnist��д����/cs231n/datasets/MNIST_data'
                           , train=True, download=False, transform=T.ToTensor())
    Val_loader = DataLoader(Mnist_val, batch_size=Batch_size)  # ,
    # sampler=ChunkSampler(NUM_VAL, NUM_TRAIN)) # ��NUM_TRAINλ�ÿ�ʼ����NUM_VAL����
    return Train_loader, Val_loader
