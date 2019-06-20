# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
# fetching data
from pathlib import Path
import requests
import pickle
import gzip
# plotting/computation/misc
import matplotlib.pyplot as plt
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)

bs = 32  # batch size
lr = 0.05  # learning rate
epochs = 5  # how many times to run through

# use CUDA (GPU) if available
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

# loss function (negative log-likelihood is this case)
loss_func = F.cross_entropy


# --- define some functions-------------------------------------
# set up dataloaders for batches (iterator that returns batches)
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs*2),
    )


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
        print("epoch", epoch, "loss:", val_loss)


# --- Sequential stuff -----------------------------------------
# reshape x to 2D single-channel 28x28 image, and also return label y
def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)

# used for applying a function (like preprocess) to each batch
class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl  # data loader
        self.func = func  # function to be applied to dl

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

# used in Sequential to apply any function
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)

'''
Replace nn.AvgPool2d with nn.AdaptiveAvgPool2d, which allows us to 
define the size of the output tensor we want, rather than the input
tensor we have. As a result, our model will work with any size input.
'''
model = nn.Sequential(
    nn.Conv2d(1, 12, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(12, 14, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(14, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

model.to(dev)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
fit(epochs, model, loss_func, opt, train_dl, valid_dl)

# --------------------------------------------------------------
# Testing the model
'''
 apply model to a dataset, ds, to obtain the predicted result on the
 first N data points. We use argmax since the index of the highest
 entry is the actual digit that is being predicted.

'''
def pred_compare(model, ds, N=None):
    N = N if N is not None else len(ds)
    preds = torch.argmax(model(preprocess(*ds[0:N])[0]), dim=1)
    y = ds[:N][-1]
    return preds, y


# accuracy: comparing our output/prediction (out) to the labels (y)
def accuracy(model, ds):
    preds, y = pred_compare(model, ds)
    wrong_ind = np.where(preds != y)  # indexes where pred is wrong
    return (preds == y).float().mean(), wrong_ind[0]

acc, wrong_ind = accuracy(model, valid_ds)
preds, y = pred_compare(model, valid_ds)
print("accuracy:", "{:0.2f}%".format(float(acc)*100))

num_img = min(8, len(wrong_ind))
fig, axes = plt.subplots(2,4)
axes = axes.flatten()
for i in range(num_img):
    if i+1 > num_img:
        break
    axes[i].imshow(valid_ds[wrong_ind[i]][0].view((28, 28)), cmap="gray")
    print(i, "Predicted: {}, Actual: {}".format(preds[wrong_ind[i]], y[wrong_ind[i]]))
plt.show()