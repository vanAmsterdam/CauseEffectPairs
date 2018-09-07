import torch
from torch.autograd import Variable

torch.manual_seed(123456)

a, b, c, d = 1, 2, -.5, 1.2
x = Variable(torch.Tensor(torch.normal(torch.zeros(100))), requires_grad = True)
y = a + b * x + c * (x**2) + d * (x**3)
z = y.detach()

y.backward()
pred = Variable(torch.Tensor(torch.zeros_like(y)), requires_grad = True)

x_in = torch.randn(3, 5, requires_grad=True)
x_target = torch.randn(3, 5)
loss = loss_fn(x_in, x_target)
loss.backward()
x_in.grad

loss_fn = torch.nn.MSELoss()
loss = loss_fn(pred, z)
loss