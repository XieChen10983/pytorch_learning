# coding=gbk
import torch
from torch import nn
import torchvision.datasets as dsets
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST
import numpy as np
# import matplotlib.pyplot as plt


class RNN(nn.Module):

    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(input_size=28,   # 也可以使用nn.RNN,但是不容易收敛。
                           hidden_size=64, num_layers=1, batch_first=True)
        # batch_first=True表示数据格式为（batch，time_step，input_size), False表示数据格式为(time_step, batch, input_size)
        self.out = nn.Linear(64, 10)

    def forward(self, INPUT):
        r_out, (_, _) = self.rnn(INPUT, None)
        out = self.out(r_out[:, -1, :])  # (batch, time_step, input_size)
        return out


rnn = RNN()

# hyper parameters
EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 28         # rnn input size / image width
LR = 0.01               # learning rate
DOWNLOAD_MNIST = False   # set to True if haven't download the data.

# training data
# train_data = dsets.MNIST(root='./mnist', train=True, transform=transforms.ToTensor, download=DOWNLOAD_MNIST)
# data = train_data.data
# target = train_data.targets
# train_data = TensorDataset(data, target)
# train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
#
# # testing data
test_data = dsets.MNIST(root='./mnist', train=False, transform=transforms.ToTensor())
transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])
train_dataset = MNIST(root='./data', train=True, transform=transformation, download=False)
test_dataset = MNIST(root='./data', train=False, transform=transformation, download=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

test_x = test_data.data.type(torch.FloatTensor)[:2000]/255.
test_y = test_data.targets.numpy().squeeze()[:2000]

optimizer = torch.optim.Adam(params=rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

if __name__ == '__main__':
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            b_x = x.view(-1, 28, 28)
            output = rnn(b_x)
            loss = loss_func(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                test_output = rnn(test_x)
                predict_y = torch.max(test_output, 1)[1].numpy().squeeze()
                accuracy = sum(np.array(predict_y) == np.array(test_y)) / test_y.size
                print('Epoch:', epoch, '| train loss: %.4f' % loss.detach().numpy())

    test_output = rnn(test_x[:10].view(-1, 28, 28))
    pred_y = torch.max(test_output, 1)[1].numpy().squeeze()
    print(pred_y, 'prediction number')
    print(test_y[:10], 'real number')
