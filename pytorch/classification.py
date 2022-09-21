import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn.functional as f
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
n_data = torch.ones(100,2)
x0 = torch.normal(2*n_data, 1)
label0 = torch.zeros(100)
x1 = torch.normal(-2*n_data, 1)
label1 = torch.ones(100)
x = torch.cat((x0,x1),0).type(torch.FloatTensor)
y  = torch.cat((label0,label1),).type(torch.LongTensor)

x,y = Variable(x), Variable(y)

class Net(torch.nn.Module):
    def __init__(self,n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.out = torch.nn.Linear(n_hidden,n_output)

    def forward(self, x):
        x = f.relu(self.hidden(x))
        x = self.out(x)
        return x


net = Net(n_feature=2, n_hidden=10, n_output=2)

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()
plt.ion()#something about plotting
for t in range(1000):
    out = net(x)

    loss = loss_func(out, y)

    optimizer.zero_grad()#所有参数梯度都降为0
    loss.backward() #反向传递
    optimizer.step() #优化梯度

    if t % 200 == 0:
        #plot and show learning %process
        plt.cla()
        prediction = torch.max(out,1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:,0], y.data.numpy()[:,1],c = pred_y,s=100,lw=0,cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(0.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()
