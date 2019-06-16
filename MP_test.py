import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from sklearn.datasets import load_iris
from torch.autograd import Variable
from torch.optim import SGD


use_cuda = torch.cuda.is_available()
print('use cuda:', use_cuda)

iris = load_iris()
print(iris.keys())

x = iris['data']
y = iris['target']
print(x.shape)
print(y)

x = torch.FloatTensor(x)
y = torch.LongTensor(y)

x, y = Variable(x), Variable(y)


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = self.predict(x)
        out = F.log_softmax(x, dim=1)
        return out


net = Net(n_feature=4, n_hidden=5, n_output=4)
print(net)

if use_cuda:
    x = x.cuda()
    y = y.cuda()
    net = net.cuda()

optimizer = SGD(net.parameters(), lr=0.5)

px, py = [], []
for i in range(1000):
    prediction = net(x)

    loss = F.nll_loss(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(i, "loss: ", loss.data)
    px.append(i)
    py.append(loss.data)

    if i % 10 == 0:
        plt.cla()
        plt.plot(px, py, 'r-', lw=1)
        plt.text(0, 0, 'Loss=%.4f' % loss.data, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)