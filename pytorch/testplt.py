import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

x = range(10)
y = range(10)
plt.plot(x, y)
plt.show()