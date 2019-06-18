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
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

bs = 64  # batch size
lr = 0.1  # learning rate
epochs = 2  # how many times to run through

# --- fetch the MNIST dataset ----------------------------------
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

# plt.imshow(x_train[0].reshape((28, 28)), cmap="gray")
# plt.show()

# --- create pytorch datasets ----------------------------------
x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
train_ds = TensorDataset(x_train, y_train)  # combine data and labels into dataset
valid_ds = TensorDataset(x_valid, y_valid)


class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # kernel is 3x3
        # stride: amount to slide kernel each step, padding: num zeros around input
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16,kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10,kernel_size=3, stride=2, padding=1)

    # xb is input batch
    def forward(self, xb):
        # input shape must be (N, Channels_in, Rows, Cols)
        xb = xb.view(-1, 1, 28, 28)  # .view is torch equiv of numpy reshape
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))

# loss function (negative log-likelihood is this case)
loss_func = F.cross_entropy

# accuracy: comparing our output/prediction (out) to the labels (yb)
def accuracy(out, yb):
    # look at largest element of each row of our predictions. That is our guess.
    preds = torch.argmax(out, dim=1)
    # see how many (porportion) of our guesses match the label
    return (preds == yb).float().mean()


# set up dataloaders for batches
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


# creates a model and an optimizer (used for updating weights & bias)
def get_model():
    model = Mnist_CNN()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return model, opt

# computes the loss for one batch of data (xb, yb)
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()  # run before training
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)  # update params in batches

        model.eval()  # run before evaluating
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(epoch, val_loss)

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)


print("accuracy on training set:", accuracy(model(x_train), y_train))
print("accuracy on validation set:", accuracy(model(x_valid), y_valid))
