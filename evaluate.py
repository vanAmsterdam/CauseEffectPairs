import torch
# import os; os.chdir("/home/wamsterd/local_scratch/git/CauseEffectPairs/")
from torch.autograd import Variable
import model.net as net
from torch.optim import SGD
from train import train
# from model.net import loss_fn
# import train
import numpy as np

torch.manual_seed(123456)

a, b, c, d = 1, 2, -.5, 1.2
x = Variable(torch.Tensor(torch.normal(torch.zeros(1000, 1))), requires_grad = True)
y = a + b * x + c * (x**2) + d * (x**3)

model = net.TwoLayerNet(1, 20, 1)

# loss_fn = net.loss_fn
optimizer = SGD(model.parameters(), lr=0.0001)

num_epochs = 1000
# train(model, num_epochs, x, y, net.loss_fn, optimizer)
train(model, num_epochs, x, y, torch.nn.MSELoss(), optimizer)
pred = model.forward(x)
print("mean mse:", torch.nn.MSELoss()(pred, x).data)

print("mean mse:", net.loss_fn(pred, x).data)
# print(np.hstack([pred.data, y.data, torch.abs(pred.data - y.data)]))

# for i in range(num_epochs):
#     pred = model(x)
#     loss = loss_fn(pred, x)
    
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     # loss.zero_grad()


train(model,
      x,
      y,
      num_epochs = 1000,
      loss = lambda x, y: torch.sum((x - y)**2),
      optimizer = SGD(lr=0.001))

pred = model.forward(x)


# loss_fn = torch.nn.MSELoss()
# loss = loss_fn(pred, y)

# def loss_fn(x, pred):
#     out = torch.sum((x - pred)**2)

#     return out

# loss_fn(y, pred)
# loss = torch.sum((y - pred)**2)
# loss.backward()

# from matplotlib import pyplot as plt
z = y.detach()

y.backward()
pred = Variable(torch.Tensor(torch.zeros_like(y)), requires_grad = True)

x_in = torch.randn(3, 5, requires_grad=True)
x_target = torch.randn(3, 5)
loss = loss_fn(x_in, x_target)
loss.backward()
x_in.grad
