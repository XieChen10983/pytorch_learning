# coding=gbk
import torch
import matplotlib.pyplot as plt

x_ = torch.linspace(-1, 1, 100)
x = torch.reshape(x_, (100, 1))
print(x)
print(x_)
y = x.pow(2) + 0.2*torch.rand(x.size())

plt.scatter(x.numpy(), y.numpy())
plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, X):
        X = torch.relu(self.hidden(X))
        return self.predict(X)


net = Net(1, 10, 1)
print(net)

plt.ion()
plt.show()

optimizer = torch.optim.SGD(params=net.parameters(), lr=0.05)
loss_func = torch.nn.MSELoss()

for t in range(1000):
    prediction = net(x)
    loss = loss_func(prediction, y)
    print(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.numpy(), y.numpy())
        print('------------------------------预测-------------------------------')
        print(prediction.detach().numpy())
        plt.plot(x.numpy(), prediction.detach().numpy(), 'r-', lw=5)
        # plt.text(0.5, 0, 'loss=%.4f' % loss[0], fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
