# coding=gbk
from torch import optim
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from time import time
import torch
from torch import nn
from PIL import Image
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset


class CatDog_DataSet(Dataset):
    def __init__(self, data_ndarray, label_ndarray, size=(224, 224)):
        self.files = data_ndarray
        self.labels = label_ndarray
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        img = np.array(Image.fromarray(self.files[item]).resize(self.size))
        label = self.labels[item]
        return img, label


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 500)
        self.fc2 = nn.Linear(500, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, INPUT):
        print(INPUT.size(), '-'*10)
        RES = torch.relu(torch.max_pool2d(self.conv1(INPUT), 2))
        print(RES.size(), '-'*20)
        RES = torch.relu(torch.max_pool2d(self.conv2_drop(self.conv2(RES)), 2))
        print(RES.size(), '-'*30)
        RES = RES.view(RES.size(0), -1)
        RES = torch.relu(self.fc1(RES))
        RES = torch.dropout(RES, train=True, p=0.2)
        RES = torch.relu(self.fc2(RES))
        RES = torch.dropout(RES, train=True, p=0.2)
        RES = self.fc3(RES)
        return torch.log_softmax(RES, dim=1)


def fit(epoch, model, optimizer, data_loader, phase='training'):
    # if phase == 'training':
    #     model.train()
    # if phase == 'validation':
    #     model.eval()
    #     volatile = True
    running_loss = 0.0
    running_correct = 0

    for batch_idx, (data, target) in enumerate(data_loader):
        print('epoch:', epoch, 'batch_index', batch_idx)
        # print(data.size())
        # print(target)
        if phase == 'training':
            optimizer.zero_grad()
        output = model(data.float())
        # print(output.size())
        loss = F.nll_loss(output, target)
        # print(loss)
        loss_func = nn.CrossEntropyLoss()
        running_loss += loss_func(output, target).detach().numpy()  # F.nll_loss(output, target, size_average=False)[0]
        preds = output.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()

    loss = running_loss/len(data_loader.dataset)
    accuracy = 100.*running_correct / len(data_loader.dataset)
    # print(f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is'
    #     f' {running_correct}/{len(data_loader.dataset)}{accuracy: {10}.{4}}')
    return loss, accuracy


catdogdset = CatDog_DataSet(root='E:\\machine learning\\datasets\\猫狗检测全\\train')
train_loader = DataLoader(catdogdset, batch_size=100, shuffle=True, num_workers=0)
test_loader = DataLoader(catdogdset, batch_size=100, shuffle=True, num_workers=0)
# for data, label in train_loader:
#     print(data.size())
#     print(data)
#     data = torch.randn_like(data)
#     print(data.size())
#     conv2d = torch.nn.Conv2d
#     conv = conv2d(3, 64, 5).double()
#     output = conv(data)
# inputt = torch.randn(32, 3, 256, 256)
# output = model(inputt)
# print(output)

# model = Net()
# # data = torch.randn(32, 3, 256, 256).float()
# # print(data)
# # print(model(data))
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
# train_losses, train_accuracy = [], []
# val_losses, val_accuracy = [], []
# for epoch in range(1, 20):
#     epoch_loss, epoch_accuracy = fit(epoch, model, optimizer, train_loader, phase='training')
#     # val_epoch_loss, val_epoch_accuracy = fit(epoch, model, optimizer, test_loader, phase='validation')
#     train_losses.append(epoch_loss)
#     train_accuracy.append(epoch_accuracy)
#     # val_losses.append(val_epoch_loss)
#     # val_accuracy.append(val_epoch_accuracy)
# #
# plt.plot(range(1, len(train_losses)+1), train_losses, 'bo', label='training')
# plt.plot(range(1, len(val_losses)+1), val_losses, 'r', label='validation loss')
# plt.legend()
# plt.show()
for data, label in train_loader:
    start = time()
    print('所花的时间为:%f秒。' % start)
