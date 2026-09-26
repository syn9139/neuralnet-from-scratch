import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
# 1: Data Collection
#   i. Import data
#   ii. Format data - separate into 28x28 (784) pics each. Every 784 _img corresponds to 1 _label
# 
# 2: Neuralnet Forwardprop
#    784(0) --> x(1) --> x(2) --> 10(out)
#   a(1) = ReLU(a(0)@W(1)) + b(1))
# 3: Neuralnet Backprop
#   i. Pick 100 random datapoints
#   ii. Calculate cost
#   iii. dC_k/dW then average across k, for all W
#   iv. dC_k/db then average across k, for all b
#   v. same with gradC
# 4: Training


## DATA COLLECTION
# Import training data from MNIST
def mnist_import(filename):
    with open('data/MNIST/raw/' + filename, 'rb') as f:
        return f.read()

mnist_settings = {
    'train-images-idx3-ubyte': {'datapoints': 60000, 'pixels': 784, 'rescale': 255, 'offset': 16},
    'train-labels-idx1-ubyte': {'datapoints': 60000, 'pixels': 1,   'rescale': 1,   'offset': 8},
    't10k-images-idx3-ubyte':  {'datapoints': 10000, 'pixels': 784, 'rescale': 255, 'offset': 16},
    't10k-labels-idx1-ubyte':  {'datapoints': 10000, 'pixels': 1,   'rescale': 1,   'offset': 8},
}

mnist_dataset_raw = {
    filename: mnist_import(filename)
    for filename in mnist_settings
}

# Shape: [neurons, datapoints]
mnist_dataset = {}
for filename, settings in mnist_settings.items():
    data = np.frombuffer(
        mnist_dataset_raw[filename],
        dtype=np.uint8,
        offset=settings['offset'],
    )
    mnist_dataset[filename] = (
        data.reshape(settings['datapoints'], settings['pixels']).T
        / settings['rescale']
    )

train_img_data = mnist_dataset['train-images-idx3-ubyte']
train_label_data = mnist_dataset['train-labels-idx1-ubyte'].squeeze().astype(int)
test_img_data = mnist_dataset['t10k-images-idx3-ubyte']
test_label_data = mnist_dataset['t10k-labels-idx1-ubyte'].squeeze().astype(int)

# Set seed
def setseed(s):
    # Reassigns the module-level training data, not local copies
    global train_img_data, train_label_data
    rng = np.random.default_rng(s)

    # Shuffle input data
    perm = rng.permutation(train_img_data.shape[1])
    train_img_data = train_img_data[:, perm]
    train_label_data = train_label_data[perm]
    return rng

seed = 2147483647
rng = setseed(seed)

##
## NEURAL NETWORK DEFINITION
##
# 784(0) --> 16(1) --> 16(2) --> 10(out)
# Recall: a(1) = ReLU(W(1)@a(0) + b(1))

# Define neuralnet special functions
def ReLU(z):
    return np.maximum(0, z)
def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))  
    return e / e.sum(axis=1, keepdims=True)


# Initialise some random 784x16 weight matrix W1 and 0 bias: dim784 to dim16
W1 = rng.standard_normal((784, 16)) * np.sqrt(2 / 784)
b1 = np.zeros((1, 16))
# Initialise some random 16x16 weight matrix W2 and 0 bias: dim16 to dim16
W2 = rng.standard_normal((16, 16)) * np.sqrt(2 / 16)
b2 = np.zeros((1, 16))
# Initialise some random 16x10 weight matrix W3 and 0 bias: dim16 to dim10
W3 = rng.standard_normal((16, 10)) * np.sqrt(2 / 16)
b3 = np.zeros((1, 10))
params = [W1, b1, W2, b2, W3, b3]

def forward(data):
    a0 = data.T
    # Layer 0 --> Layer 1
    z1 = data.T @ W1 + b1
    a1 = ReLU(z1)
    # Layer 1 --> Layer 2
    z2 = a1 @ W2 + b2
    a2 = ReLU(z2)
    # Layer 2 --> Layer Out
    z3 = a2 @ W3 + b3
    a3 = softmax(z3)
    return a0, z1, a1, z2, a2, z3, a3


## BACKPROPAGATION

def backprop(x):
    # Pick 100 datapoints, starting from the beginning of the training set
    subset_img = train_img_data[:, x*100 : x*100 + 100]
    subset_label = train_label_data[x*100 : x*100 + 100]
    # Convert labels to a 100x10 one-hot matrix
    desired_output = np.zeros((100, 10))
    desired_output[np.arange(100), subset_label] = 1
    # Import current neuron values
    a0, z1, a1, z2, a2, z3, a3 = forward(subset_img)

    # dC_0/dW = dC_0/da da/dz dz/dW || Keep in mind that these are all vectors with 100 derivatives
    #         =    delta_i    dz/dW
    # Output --> Layer 3
    delta3 = a3 - desired_output
    dz_dW3 = a2.T
    dC_dW3 = dz_dW3 @ delta3 / 100

    # Layer 3 --> Layer 2
    delta2 = delta3 @ W3.T * np.heaviside(z2, 0)
    dz_dW2 = a1.T
    dC_dW2 = dz_dW2 @ delta2 / 100

    # Layer 2 --> Layer 1
    delta1 = delta2 @ W2.T * np.heaviside(z1, 0)
    dz_dW1 = a0.T
    dC_dW1 = dz_dW1 @ delta1 / 100

    # gradients
    grads = [
        dC_dW1, np.mean(delta1, axis=0, keepdims=True),
        dC_dW2, np.mean(delta2, axis=0, keepdims=True),
        dC_dW3, np.mean(delta3, axis=0, keepdims=True),
    ]
    return grads

# SGD
if __name__ == "__main__":
    for epoch in range(10):
        setseed(seed + epoch) # shuffle data
        for i in range(600): # 600 x 100 training pictures
            grads = backprop(i)
            for p, grad in zip(params, grads):
                p -= 0.1 * grad
 
    np.savez(
        'weights.npz',
        W1=W1, b1=b1,
        W2=W2, b2=b2,
        W3=W3, b3=b3,
    )
    print("Training done, weights saved to weights.npz")
