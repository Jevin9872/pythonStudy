import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as f
from torch.autograd import Variable
import matplotlib.pyplot as plt

#fake data
x = torch.linspace(-5,5,200) #x data(tensor), shape=(100,1)
x = Variable(x)
x_np = x.data.numpy()

y_relu = f.relu(x).data.numpy()
y_sigmoid = torch.sigmoid(x).data.numpy()
y_tanh = torch.tanh(x).data.numpy()

plt.figure(1,figsize=(8,6))
plt.subplot(221)
plt.plot(x_np,y_relu, c='red',label='relu')
plt.ylim((-1,5))
plt.legend(loc='best')

plt.figure(1,figsize=(8,6))
plt.subplot(222)
plt.plot(x_np,y_sigmoid, c='red',label='sigmoid')
plt.ylim((-0.2,1.2))
plt.legend(loc='best')

plt.figure(1,figsize=(8,6))
plt.subplot(223)
plt.plot(x_np,y_tanh, c='red',label='tanh')
plt.ylim((-1.3,1.3))
plt.legend(loc='best')
plt.show()

