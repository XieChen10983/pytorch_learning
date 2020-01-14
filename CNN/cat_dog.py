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
    def __init__(self, root, size=(256, 256)):
        # self.files = data_ndarray
        # self.labels = label_ndarray
        self.root = root
        self.size = size
        self.file_absolute_names = [self.root + '\\' + name for name in os.listdir(self.root)]

    def __len__(self):
        file_names = os.listdir(self.root)
        return len(file_names)

    def __getitem__(self, item):

        img = np.transpose(np.array(Image.open(self.file_absolute_names[item]))/255., (2, 0, 1))
        labels = [file_absolute_name.split('\\')[-1].split('.')[0] for file_absolute_name in self.file_absolute_names]
        label = labels[item]
        return img, label


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(56180, 500)
        self.fc2 = nn.Linear(500, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, INPUT):
        RES = torch.relu(torch.max_pool2d(self.conv1(INPUT), 2))
        RES = torch.relu(torch.max_pool2d(self.conv2_drop(self.conv2(RES)), 2))
        RES = RES.view(RES.size(0), -1)
        RES = torch.relu(self.fc1(RES))
        RES = torch.dropout(RES, train=True, p=0.2)
        RES = torch.relu(self.fc2(RES))
        RES = torch.dropout(RES, train=True, p=0.2)
        RES = self.fc3(RES)
        return torch.log_softmax(RES, dim=1)


def gen_numpy_catsanddogs(in_path='E:/machine learning/deep learning/pytorch/CNN/reshaped'):
    Label = {'cat': 0, 'dog': 1}
    print('正在生成猫狗的numpy数据...')
    name_list = os.listdir(in_path)
    name_list = [in_path + '/' + name for name in name_list]
    all_data = []
    num_sum = len(name_list)
    for index, name in enumerate(name_list):
        img = np.transpose(np.array(Image.open(name)), (2, 0, 1))
        img_data = img.reshape((1, -1))
        cat_or_dog = name.split('/')[-1].split('.')[0]
        img_data = np.hstack((img_data, np.array(Label[cat_or_dog]).reshape((1, -1))))
        all_data.append(img_data)
        if index % 500 == 0:
            print('[' + '=' * 2 * int(index / 500) + '>' + '%f' % (index / num_sum * 100) + '%' + ']')
    all_data = np.array(all_data)
    print('猫狗的numpy数据生成完毕！')
    print(all_data.shape)


# gen_numpy_catsanddogs()


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
        output = model(data)
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


#
catdogdset = CatDog_DataSet(root='E:\\machine learning\\deep learning\\pytorch\\CNN\\reshaped')
train_loader = DataLoader(catdogdset, batch_size=100, shuffle=True, num_workers=0)
# test_loader = DataLoader(catdogdset, batch_size=100, shuffle=True, num_workers=0)
for data, label in train_loader:
    print(data.size())
    start = time()
    print(start)
#     print(data)
#     data = torch.randn_like(data)
#     print(data.size())
#     conv2d = torch.nn.Conv2d
#     conv = conv2d(3, 64, 5).double()
#     output = conv(data)
#
# #     print(output.size())
# #     data = data.detach().numpy()
# #     data1 = np.transpose(data[1], (1, 2, 0))
# #     plt.imshow(data1)
# #     plt.title(label[1])
# #     plt.show()
# # # path = 'E:\\machine learning\\datasets\\猫狗检测全\\train'
# # l = os.listdir(path)
# # name = [path + '\\' + data for data in l]
# # label = [name.split('\\')[-1].split('.')[0] for name in name]
# # print(name)
# # start = time()
# # # file = np.array(Image.open(name[0]).resize((224, 224)))
# # # plt.imshow(file)
# # # print(file.shape)
# # # plt.show()
# # files = [np.array(Image.open(name).resize((224, 224))) for name in name]
# # end = time()
# # print(label)
# # print(len(name), len(label))
# # files = np.array(files)
# # print(files.shape)
# # print('时间：', end-start)
#
# #
# # model = Net()
# # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
# # train_losses, train_accuracy = [], []
# # val_losses, val_accuracy = [], []
# # for epoch in range(1, 20):
# #     epoch_loss, epoch_accuracy = fit(epoch, model, optimizer, train_loader, phase='training')
# #     val_epoch_loss, val_epoch_accuracy = fit(epoch, model, optimizer, test_loader, phase='validation')
# #     train_losses.append(epoch_loss)
# #     train_accuracy.append(epoch_accuracy)
# #     val_losses.append(val_epoch_loss)
# #     val_accuracy.append(val_epoch_accuracy)
# # #
# # plt.plot(range(1, len(train_losses)+1), train_losses, 'bo', label='training')
# # plt.plot(range(1, len(val_losses)+1), val_losses, 'r', label='validation loss')
# # plt.legend()
# # plt.show()
#
