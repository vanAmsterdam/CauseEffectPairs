from torch.tensor import Tensor
from tqdm import tqdm
# from torch.nn import NeuralNet
# from torch.nn.loss import Loss, MSE
# from torch.optim import Optimizer, SGD
# from data import DataIterator, BatchIterator



def train(model, num_epochs, x, target, loss_fn, optimizer, gradient_clip = None):
    for i in tqdm(range(num_epochs)):
        pred = model(x)
        loss = loss_fn(pred, target)
        
        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        if not gradient_clip is None:
            for param in model.parameters():
                param.grad.clamp_(*gradient_clip)


        optimizer.step()
        # loss.zero_grad()


# def train(net: 1,
#           inputs: Tensor,
#           targets: Tensor,
#           num_epochs: int = 5000,
#           iterator: DataIterator = BatchIterator(),
#           loss: 1,
#           optimizer: Optimizer = SGD()) -> None:
#     for epoch in range(num_epochs):
#         epoch_loss = 0.0
#         for batch in iterator(inputs, targets):
#             predicted = net.forward(batch.inputs)
#             epoch_loss += loss.loss(predicted, batch.targets)
#             grad = loss.grad(predicted, batch.targets)
#             net.backward(grad)
#             optimizer.step(net)
#         print(epoch, epoch_loss)


