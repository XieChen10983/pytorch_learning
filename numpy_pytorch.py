import torch
import torch.nn.functional as fntnl
# from torch.autograd import Variable
import matplotlib.pyplot as plt

# data
x_np = torch.linspace(-5, 5, 200)
# X = Variable(x)
# x_np = x.data.numpy()

# y_relu = fntnl.relu(x).numpy()
# y_sigmoid = fntnl.sigmoid(x).numpy()
# y_tanh = fntnl.tanh(x).numpy()
# y_softplus = fntnl.softplus(x).numpy()
# y_softmax = fntnl.softmax(x)
y_relu = torch.relu(x_np).numpy()
y_sigmoid = torch.sigmoid(x_np).numpy()
y_tanh = torch.tanh(x_np).numpy()
y_softplus = fntnl.softplus(x_np).numpy()

plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()
