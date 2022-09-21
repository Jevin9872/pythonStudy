import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from torch.autograd import Variable
import torch.nn.functional as f
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-3,3,100),dim=1)
y = x.pow(2)+4*torch.rand(x.size())
# x,y = Variable(x), Variable(y)


class Net(torch.nn.Module):
    def __init__(self,n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)

    def forward(self, x):
        x = f.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(n_feature=1, n_hidden=10, n_output=1)

optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
loss_func = torch.nn.MSELoss()
plt.ion()#something about plotting
for t in range(1000):
    prediction = net(x)
    # print(prediction)

    loss = loss_func(prediction, y)

    optimizer.zero_grad()#所有参数梯度都降为0
    loss.backward() #反向传递
    optimizer.step() #优化梯度

    if t % 200 == 0:
        #plot and show learning %process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 1, 'Loss=%.4f' % loss.data, fontdict={'size':20, 'color':'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()