"""
- Neural networks can be constructed using the torch.nn package
- nn depends on autograd to define models and differentiate them
- covnet = CNN = Convolutional Neural Network

A typical training procedure for a neural network is as follows:
1) Define the neural network that has some learnable parameters (or weights)
2) Iterate over a dataset of inputs
3) Process input through the network
4) Compute the loss (how far is the output from being correct)
5) pPopagate gradients back into the network’s parameters
6) Update the weights of the network, typically using a simple update rule: 
       weight = weight - learning_rate * gradient

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

"""
Loss functions
- A loss function takes the (output, target) pair of inputs, and computes a value that estimates how far away the output is from the target.
- A simple loss is: nn.MSELoss which computes the mean-squared error between the input and the target.
"""
output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
# tensor(0.9914, grad_fn=<MseLossBackward>)
# <MseLossBackward object at 0x7f9eb40c88d0>
# <AddmmBackward object at 0x7f9eb40c85f8>
# <AccumulateGrad object at 0x7f9eb40c85f8>

"""
Backpropagation
- To backpropagate the error all we have to do is to loss.backward()
- Need to clear the existing gradients though, else gradients will be accumulated to existing gradients
"""
net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# conv1.bias.grad before backward
# tensor([0., 0., 0., 0., 0., 0.])
# conv1.bias.grad after backward
# tensor([-0.0205,  0.0088,  0.0135,  0.0123,  0.0098, -0.0036])

"""
Update the weights
- The simplest update rule used in practice is the Stochastic Gradient Descent (SGD)
    weight = weight - learning_rate * gradient
- However, as you use neural networks, you want to use various different update rules such as SGD, Nesterov-SGD, Adam, RMSProp, etc
- torch.optim implements all these methods
"""

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
