import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader  # batches and shuffles the data
from torchvision import datasets, transforms

# 1: Data Collection
# 2: Neuralnet Forwardprop
#    784(0) --> 16(1) --> 16(2) --> 10(out)
#   a(1) = ReLU(a(0)@W(1) + b(1))
# 3: Neuralnet Backprop
# 4: Training

# Load data from MNIST
def load_mnist(train):
    ds = datasets.MNIST("data", train=train, download=True)
    image = ds.data.float().view(-1, 784) / 255.0
    label = ds.targets
    return image, label

train_image, train_label = load_mnist(True) 
test_image, test_label = load_mnist(False)

# Define the MLP forward pass w/ Kaiming initialisation
W1 = (torch.randn(784, 16) * np.sqrt(2 / 784))
b1 = torch.zeros(16)
W2 = (torch.randn(16, 16) * np.sqrt(2 / 16))
b2 = torch.zeros(16)
W3 = (torch.randn(16, 10) * np.sqrt(2 / 16))
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
    logits = forward_pass(data)
    output = torch.sigmoid(logits)
    # reset previous gradients
    for p in params: 
        p.grad = None

    # create one-hot expectation tensor
    expected = F.one_hot(train_label[x*batch:x*batch+batch], num_classes=10).float()
    loss = ((output - expected)**2).sum(dim=1).mean()
    loss.backward()


    for p in params:
        p.data += -0.1 * p.grad 

# Do SGD
batch = 100
if __name__ == "__main__":
    for _ in range(10):
        for x in range(600):
            backprop(train_image[x*batch:x*batch+batch], train_label[x*batch:x*batch+batch])

    # Save the weights to a file
    np.savez('weights_with_pytorch.npz', *[p.detach().cpu().numpy() for p in params])
    print("Training done, weights saved to weights_with_pytorch.npz")
