import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import requests
import pickle
import gzip
import matplotlib.pyplot as plt
import numpy as np

# we must first fetch the MNIST dataset
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

# if not already downloaded, do so now.
if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

# unpickle/unzip file
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

# see an example of the data.
# Note: image is string of 784 entries. Reshape to 28x28 array.
# plt.imshow(x_train[0].reshape((28, 28)), cmap="gray")
# plt.show()

# convert data to pytorch tensors
x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))

n,c = x_train.shape
sqrtc = int(np.sqrt(c))

# create 3D tensors: (N, 28, 28)
x_train, x_valid = map(lambda x: x.reshape(len(x), sqrtc, sqrtc), (x_train, x_valid))

weights = torch.randn(sqrtc, sqrtc, 10) / sqrtc
weights.requires_grad_()  # enable autograd tracking of the weights
bias = torch.zeros(10, requires_grad=True)

# activation function
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

# forward computation from input to first layer
def model(xb):
    # torch does not have tensordot. Use einsum instead.
    # activation_fcn(xb @ weights + bias)
    return log_softmax(torch.einsum('ijk,jkl->il', xb, weights) + bias)

# loss function (negative log-likelihood is this case)
def loss_func(input, target):
    # target.shape[0] is length of labels array
    # input is of shape (N, 10). Of the 10, look at index of the label to see
    #   how large the element is there. Average over all such elements.
    return -input[range(target.shape[0]), target].mean()

# accuracy: comparing our output/prediction (out) to the labels (yb)
def accuracy(out, yb):
    # look at largest element of each row of our predictions. That is our guess.
    preds = torch.argmax(out, dim=1)
    # see how many (porportion) of our guesses match the label
    return (preds == yb).float().mean()

bs = 64  # batch size
lr = 0.5  # learning rate
epochs = 2  # how many times to run through

for epoch in range(epochs):
    for i in range((n-1) // bs + 1):
        xb = x_train[i*bs: (i+1)*bs]
        yb = y_train[i*bs: (i+1)*bs]
        preds = model(xb)
        loss = loss_func(preds, yb)

        loss.backward()  # backprop (autograd) calculates grads
        with torch.no_grad():
            weights -= weights.grad * lr  # update weights
            bias -= bias.grad * lr  # update bias
            weights.grad.zero_()  # reset weights.grad in-place
            bias.grad.zero_()  # reset bias.grad in-place


print("loss", loss_func(model(xb), yb))
print("accuracy", accuracy(model(xb), yb))

print("accuracy on validation set", accuracy(model(x_valid), y_valid))
