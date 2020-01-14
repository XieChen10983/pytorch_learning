# coding=gbk
import torch
import torch.nn as nn
from keras.datasets import mnist
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torch import optim
import matplotlib.pyplot as plt


# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)
#
#     def forward(self, x):
#         x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
#         x = torch.relu(torch.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = torch.relu(self.fc1(x))
#         x = torch.dropout(x, train=True, p=0.1)
#         x = self.fc2(x)
#         return torch.log_softmax(x, dim=0)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=25, out_channels=50, kernel_size=3, padding=1)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=2)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=2)
        self.linear1 = nn.Linear(in_features=7*7*50, out_features=1000)
        self.linear2 = nn.Linear(in_features=1000, out_features=100)
        self.linear3 = nn.Linear(in_features=100, out_features=10)
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.4)
        self.dropout2d = nn.Dropout2d(0.2)

    def forward(self, INPUT):
        RES = self.conv1(INPUT)
        RES = torch.relu(self.maxpooling1(RES))
        RES = torch.relu(self.maxpooling2(self.conv2(RES)))
        RES = self.dropout2d(RES)
        RES = RES.view(-1, 7*7*50)
        RES = self.linear1(RES)
        RES = torch.relu(RES)
        RES = self.dropout1(RES)
        RES = torch.relu(self.linear2(RES))
        RES = self.dropout2(RES)
        RES = self.linear3(RES)
        return torch.log_softmax(RES, dim=0)


class MNIST_DataSet(Dataset):
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


def train(EPOCH, MODEL, DATA_LOADER, OPTIMIZER):
    print('----------------------------------------开始训练...-------------------------------------------')
    losses, accuracies = [], []
    for EPO in range(1, EPOCH + 1):
        print('---------------------------------------EPOCH%d------------------------------------------' % EPO)
        running_loss = 0.
        running_correct = 0.
        for Batch_index, (Data, Target) in enumerate(DATA_LOADER):
            print('-------------------------------------BATCH%d----------------------------------------' % Batch_index)
            Data = Data.view(-1, 1, 28, 28)
            Target = Target.long()
            # print(Data.size())
            # print(Target.size())
            OPTIMIZER.zero_grad()
            Output = MODEL(Data)
            Output_numpy = Output.detach().numpy()
            predict_target = torch.from_numpy(np.argmax(Output_numpy, axis=1))
            # print(predict_target)
            Loss_function = torch.nn.CrossEntropyLoss()
            Loss = Loss_function(Output, Target)
            Loss.backward()
            OPTIMIZER.step()
            running_loss += Loss
            # predict_target = Output.max(dim=1)
            # running_correct += predict_target.eq(Target.view_as(predict_target)).sum()
            running_correct += np.sum(predict_target.numpy() == Target.numpy())
            # print(predict_target)
            # print(Target)
        loss = running_loss / len(DATA_LOADER.dataset)
        accuracy = 100. * running_correct / len(DATA_LOADER.dataset)
        losses.append(loss)
        accuracies.append(accuracy)
    print(losses)
    print(accuracies)
    plt.figure(1, figsize=(8, 6))
    plt.scatter(np.arange(1, len(losses)+1), y=np.array(losses), c='r', s=100)
    plt.scatter(np.arange(1, len(accuracies)+1), y=np.array(accuracies), c='b', s=50)
    plt.show()


data = mnist.load_data()
training_data = data[0][0] / 255.
print(training_data.shape)
training_label = data[0][1]
testing_data = data[1][0] / 255.
testing_label = data[1][1]


training_dset = MNIST_DataSet(training_data, training_label, size=(28, 28))
train_loader = DataLoader(training_dset, batch_size=32, shuffle=True, num_workers=0)
testing_dset = MNIST_DataSet(testing_data, testing_label, size=(28, 28))
testing_loader = DataLoader(testing_dset, batch_size=32, shuffle=True, num_workers=0)

net = Net()
optimizer = optim.Adam(params=net.parameters(), lr=0.01, weight_decay=0.01)

# 下面开始训练。。。
if __name__ == '__main__':
    # for Data, Target in train_loader:
    #     print(Data.size())
    #     print(Target.size())
    #     pass
    train(EPOCH=20, MODEL=net, DATA_LOADER=train_loader, OPTIMIZER=optimizer)
    train_loss, train_accuracy = [], []
    val_loss, val_accuracy = [], []
    print(len(train_loader.dataset))
