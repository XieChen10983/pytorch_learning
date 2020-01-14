# coding=gbk
import torch
from tqdm import tqdm
from torch import nn


class Auto_encoder(nn.Module):
    def __init__(self, feature_dim=784, latent_dim=1000):
        super(Auto_encoder, self).__init__()
        self.encoder = nn.Linear(in_features=feature_dim, out_features=latent_dim)
        self.decoder = nn.Linear(in_features=latent_dim, out_features=feature_dim)

    def forward(self, Input: torch.Tensor):
        code = self.encoder(Input)
        decode = self.decoder(code)
        return code, decode


class Sparse_Auto_Encoder:
    def __init__(self, Input, AE):
        self.input = Input
        self.auto_encoder = AE
        self.optimizer = torch.optim.Adam(params=AE.parameters(), lr=0.002, betas=(0.5, 0.999))
        self.epochs = 10
        self.current_epoch = 0

    def train(self):
        for epoch in tqdm(range(self.epochs)):
            print('hi')
            self.current_epoch = epoch + 1
            self.train_one_epoch()
        _, out = self.auto_encoder(self.input)
        # print(out)
        return out

    def train_one_epoch(self):
        hidden, output = self.auto_encoder(self.input)
        loss = self.cal_loss(self.input, output, hidden)
        self.auto_encoder.zero_grad()
        loss.backward()
        self.optimizer.step()

    @staticmethod
    def cal_loss(Input, Output, Hidden):
        loss1 = torch.sum(torch.pow((Input - Output), 2))
        loss2 = torch.sum(torch.pow(Hidden, 2))
        loss = loss2 + loss1
        return loss1


import cv2
import numpy as np
from PIL import Image
image = Image.open(fp='B:/1.jpg')
image = image.resize((128, 128))
image = torch.from_numpy(np.array(image))
# image = torch.from_numpy(cv2.imread(filename='B:/1.jpg'))
image = torch.reshape(image, shape=(1, -1)).float()
print(image.shape)
ae = Auto_encoder(feature_dim=image.size(1), latent_dim=10000)
# ae.encoder.weight.data = torch.ones(size=(10000, 49152))
ae.encoder.weight.data.fill_(0)
print(ae.encoder.weight)
sae = Sparse_Auto_Encoder(Input=image, AE=ae)
# out = sae.train()
# print(out.shape)
# out = out.reshape((128, 128, 3)).detach().numpy()
# # image = Image.fromarray(out)
# # image.show(title='hi')
# cv2.imshow('nihao', out.astype(np.uint8))
# cv2.waitKey()
# cv2.destroyAllWindows()
# print(sae)
