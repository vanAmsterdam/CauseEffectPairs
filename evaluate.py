import torch
from importlib import reload
# import os; os.chdir("/home/wamsterd/local_scratch/git/CauseEffectPairs/")
from torch.autograd import Variable
from torch.optim import SGD
import matplotlib.pyplot as plt
import numpy as np

import train; reload(train)
import model.net as net; reload(net)

torch.manual_seed(123456)
n_units = int(10e4)
a, b, c, d = 1, 2, -.5, 1.2
x = Variable(torch.Tensor(np.linspace(-2, 2, n_units).reshape(n_units, 1)), requires_grad = True)
y = a + b * x + c * (x**2) + d * (x**3)


# plot data 
plt.scatter(x.detach().numpy(), y.detach().numpy())
plt.show()

model = net.TwoLayerNet(1, 40, 1)
# model = net.ThreeLayerNet(1, 20, 20, 1)
# for param in model.parameters():
#       print(param)
#       param0 = param

# loss_fn = net.loss_fn
optimizer = SGD(model.parameters(), lr=0.001)#, momentum=0.9)#, weight_decay=.1)

num_epochs = 1000
# train(model, num_epochs, x, y, net.loss_fn, optimizer)
train.train(model, num_epochs, x, y, 
      torch.nn.MSELoss(), 
      optimizer, 
      gradient_clip = (-10, 10))

pred = model.forward(x)
print("mean mse:", torch.nn.MSELoss()(pred, x).data)

plt.scatter(x.detach().numpy(), y.detach().numpy())
plt.scatter(x.detach().numpy(), pred.detach().numpy(), c = "red")
plt.show()

