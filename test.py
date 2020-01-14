# coding=gbk
from torch import optim
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from keras.datasets import mnist
from torchvision.datasets import MNIST
import numpy as np
from PIL import Image
# import numpy as np


class MNIST_DataSet(Dataset):
    def __init__(self, data_ndarray, label_ndarray, size=(224, 224)):
        self.files = data_ndarray
        self.labels = label_ndarray
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        img = np.array(Image.fromarray(self.files[item]).resize(self.size)).reshape((1, self.size[0], self.size[1]))
        label = self.labels[item]
        return img, label


# conv1 = nn.Conv2d(1, out_channels=10, kernel_size=3, padding=1)

transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])
train_dataset = MNIST(root='./data', train=True, transform=transformation, download=False)
# print(train_dataset.size())
test_dataset = MNIST(root='./data', train=False, transform=transformation, download=False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
print(train_loader)
print(test_loader)
# for batch_idx, (data, target) in enumerate(train_loader):
#     # print('epoch:', epoch, 'batch_index', batch_idx)
#     print(data)
#     RES = conv1(data)
#     # print(target.size())

data = mnist.load_data()
training_data = data[0][0] / 255.
# print(training_data.shape)
training_label = data[0][1]
testing_data = data[1][0] / 255.
testing_label = data[1][1]


training_dset = MNIST_DataSet(training_data, training_label, size=(28, 28))
train_loader = DataLoader(training_dset, batch_size=32, shuffle=True, num_workers=0)
testing_dset = MNIST_DataSet(testing_data, testing_label, size=(28, 28))
test_loader = DataLoader(testing_dset, batch_size=32, shuffle=True, num_workers=0)
print(train_loader)
print(test_loader)
# if __name__ == '__main__':
#     for _ in range(10):
#         print('--------------------------------------------------------------------------------------')
#     for batch_idx, (data, target) in enumerate(train_loader):
#         print('--------------------------------------batch_index %d----------------------------------' % batch_idx)
#         # print('epoch:', epoch, 'batch_index', batch_idx)
#         print(data)
#         RES = conv1(data)
#         # print(target.size())


# def plot_img(image):
#     image = image.numpy()[0]
#     mean = 0.1307
#     std = 0.3081
#     image = ((mean * image) + std)
#     plt.imshow(image, cmap='gray')
#     plt.show()
#
#
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = torch.dropout(x, train=True, p=0.1)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=0)


def fit(epoch, model, optimizer, data_loader, phase='training', volatile=False):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile = True
    running_loss = 0.0
    running_correct = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        target = target.long()
        # print('epoch:', epoch, 'batch_index', batch_idx)
        print(data.size())
        print(target.size())
        # conv1 = nn.Conv2d(1, out_channels=10, kernel_size=3, padding=1)
        # RES = conv1(data)
        # print(RES)
        # print('------------------------------------------%d---------------------------------' % batch_idx)
        if phase == 'training':
            optimizer.zero_grad()
        output = model(data)
        print(output.size())
        print(target)
        loss = F.nll_loss(output, target)
        print(loss)
        loss_func = nn.CrossEntropyLoss()
        running_loss += loss_func(output, target).detach().numpy()  # F.nll_loss(output, target, size_average=False)[0]
        preds = output.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()
    loss = running_loss/len(data_loader.dataset)
    accuracy = 100.*running_correct / len(data_loader.dataset)
    #print(f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is'
    #     f' {running_correct}/{len(data_loader.dataset)}{accuracy: {10}.{4}}')
    return loss, accuracy


if __name__ == '__main__':
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    train_losses, train_accuracy = [], []
    val_losses, val_accuracy = [], []
    fit(2, model, optimizer, train_loader, phase='training')
    for epoch in range(1, 20):
        epoch_loss, epoch_accuracy = fit(epoch, model, optimizer, train_loader, phase='training')
        val_epoch_loss, val_epoch_accuracy = fit(epoch, model, optimizer, test_loader, phase='validation')
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)

    plt.plot(range(1, len(train_losses)+1), train_losses, 'bo', label='training')
    plt.plot(range(1, len(val_losses)+1), val_losses, 'r', label='validation loss')
    plt.legend()
    plt.show()
