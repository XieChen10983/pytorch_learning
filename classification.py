# coding=gbk
import torch
import matplotlib.pyplot as plt


class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, X):
        X = torch.relu(self.hidden(X))
        return self.predict(X)


n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data, 1)
y1 = torch.ones(100)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1)).type(torch.LongTensor)

net = Net(2, 10, 2)
plt.ion()
plt.show()

optimizer = torch.optim.SGD(params=net.parameters(), lr=0.002)
loss_func = torch.nn.CrossEntropyLoss()

for t in range(500):
    out = net(x)
    loss = loss_func(out, y)
    print(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 2 == 0:
        plt.cla()
        # plt.scatter(x.numpy(), y.numpy())
        print('------------------------------预测-------------------------------')
        print(out.detach().numpy())
        prediction = torch.max(torch.softmax(out, dim=1), 1)[1]
        pred_y = prediction.numpy().squeeze()
        target_y = y.numpy()
        plt.scatter(x.numpy()[:, 0], x.numpy()[:, 1], c=pred_y, s=100, lw=0)
        # plt.plot(x.numpy(), prediction.detach().numpy(), 'r-', lw=5)
        # plt.text(0.5, 0, 'loss=%.4f' % loss[0], fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()

plt.figure(2)
plt.scatter(x.numpy()[:, 0], x.numpy()[:, 1], c=y, s=100, lw=0)
plt.show()

