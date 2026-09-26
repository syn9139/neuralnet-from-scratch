from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets # batches and shuffles the data

# 1: Data Collection
# 2: Neuralnet Forwardprop
#    784(0) --> x(1) --> x(2) --> 47(out)
#   a(1) = ReLU(a(0)@W(1) + b(1)
# 3: Neuralnet Backprop
# 4: Training

# Load data from EMNIST (balanced: 47 classes, digits + letters)
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
def load_emnist(train):
    ds = datasets.EMNIST(DATA_DIR, split='balanced', train=train, download=True)
    image = ds.data.transpose(1, 2).float().reshape(-1, 784) / 255.0  # EMNIST is stored transposed
    label = ds.targets
    return image, label, ds.classes

train_image, train_label, classes = load_emnist(True) 
test_image, test_label, _ = load_emnist(False)
n_classes = len(classes)
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
W1 = (torch.randn(784, 128, generator=g) * np.sqrt(2 / 784))
b1 = torch.zeros(128)
W2 = (torch.randn(128, 128, generator=g) * np.sqrt(2 / 128))
b2 = torch.zeros(128)
W3 = (torch.randn(128, n_classes, generator=g) * np.sqrt(2 / 128))
b3 = torch.zeros(n_classes)

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

