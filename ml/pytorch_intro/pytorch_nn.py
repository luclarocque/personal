import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''
torch.nn depends on autograd to define models and differentiate them.
An nn.Module contains layers, and a method forward(input) that returns the output.

https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
'''
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # we need only define the forward function (backward comes from autograd)
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
# print(net)

params = list(net.parameters())  # all learnable parameters
# print(len(params))
# print(params[0].size())  # conv1's .weight

# try a random 32x32 input
input = torch.randn(1, 1, 32, 32)
out = net(input)
# print(out)


# Zero the gradient buffers of all parameters and backprops with random gradients:
net.zero_grad()
out.backward(torch.randn(1, 10))

'''
NOTE
torch.nn only supports mini-batches. The entire torch.nn package only
supports inputs that are a mini-batch of samples, and not a single sample.
For example, nn.Conv2d will take in a 4D Tensor of
nSamples x nChannels x Height x Width.
If you have a single sample, just use input.unsqueeze(0) to add a fake batch dimension.
'''

'''
A loss function takes the (output, target) pair of inputs, and computes a value
that estimates how far away the output is from the target.
There are several different loss functions under the nn package.
simple loss: nn.MSELoss computes the mean-squared error between the input and the target.
'''

output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
# print(loss)

'''
So, when we call loss.backward(), the whole graph is differentiated w.r.t. the loss,
and all Tensors in the graph that has requires_grad=True will have their
.grad Tensor accumulated with the gradient.

To backpropagate the error all we have to do is loss.backward().
You need to clear the existing gradients though,
else gradients will be accumulated to existing gradients.
'''
net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


'''
Final step: update the weights
The simplest update rule used in practice is the Stochastic Gradient Descent (SGD)
  weight = weight - learning_rate * gradient
'''
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

'''
However, as you use neural networks, you want to use various different update rules
such as SGD, Nesterov-SGD, Adam, RMSProp, etc.
To enable this, use: torch.optim that implements all these methods.
'''
# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update

