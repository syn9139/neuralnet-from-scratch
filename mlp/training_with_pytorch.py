from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets # batches and shuffles the data

# 1: Data Collection
# 2: Neuralnet Forwardprop
#    784(0) --> 16(1) --> 16(2) --> 10(out)
#   a(1) = ReLU(a(0)@W(1) + b(1)
# 3: Neuralnet Backprop
# 4: Training

# Load data from MNIST
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
def load_mnist(train):
    ds = datasets.MNIST(DATA_DIR, train=train, download=True)
    image = ds.data.float().view(-1, 784) / 255.0
    label = ds.targets
    return image, label

train_image, train_label = load_mnist(True) 
test_image, test_label = load_mnist(False)
train_image0, train_label0 = train_image, train_label

# Set seed
def setseed(s):
    global train_image, train_label
    g = torch.Generator().manual_seed(s)

    # Shuffle input data
    perm = torch.randperm(train_image0.shape[0], generator=g)
    train_image = train_image0[perm]
    train_label = train_label0[perm]
    return g

seed = 2147483647
g = setseed(seed)

# Define the MLP forward pass w/ Kaiming initialisation
W1 = (torch.randn(784, 16, generator=g) * np.sqrt(2 / 784))
b1 = torch.zeros(16)
W2 = (torch.randn(16, 16, generator=g) * np.sqrt(2 / 16))
b2 = torch.zeros(16)
W3 = (torch.randn(16, 10, generator=g) * np.sqrt(2 / 16))
b3 = torch.zeros(10)

params = [W1, b1, W2, b2, W3, b3]
for p in params:
    p.requires_grad = True

def forward_pass(data):
    h1 = torch.relu(data @ W1 + b1)
    h2 = torch.relu(h1 @ W2 + b2)
    logits = h2 @ W3 + b3
    return logits

def backprop(data, labels): 
    # reset previous gradients
    for p in params: 
        p.grad = None

    logits = forward_pass(data)
    loss = F.cross_entropy(logits, labels)
    
    loss.backward()

# Do SGD
batch = 100
lr = 0.1
if __name__ == "__main__":
    for epoch in range(10):
        setseed(seed + epoch) # shuffle data
        for x in range(600):
            backprop(train_image[x*batch:x*batch+batch], train_label[x*batch:x*batch+batch])
            for p in params:
                with torch.no_grad(): p -= lr * p.grad

    # Save the weights to a file
    np.savez('weights_with_pytorch.npz', *[p.detach().cpu().numpy() for p in params])
    print("Training done, weights saved to weights_with_pytorch.npz")
