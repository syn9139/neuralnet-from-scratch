# neuralnet-from-scratch

A lightweight handwritten character recogniser. `mlp` is built without any ML libraries like PyTorch or TensorFlow. `optimisers` implements SGD, momentum, and AdamW by hand. Loosely based on 3blue1brown's neural network series. Data trained on MNIST/EMNIST.

## Features

- Neural network built from scratch using only NumPy without ML libraries.
- Trains on the full MNIST/EMNIST dataset.
- Forward pass, loss and back propagation derived manually
- 784 → 16 → 16 → 10 MLP with ReLU hidden layers.
- Softmax output layer with cross-entropy loss.
- Predictions visualised via matplotlib, showing a test digit and neural network prediction side by side.
- The same neural network but with PyTorch for reference.

## Results

Mean + standard deviation on the 10,000-image MNIST test set using 10 runs:

| Implementation       |   Accuracy |
| -------------------- | ---------: |
| NumPy (from scratch) | **(95.00 ± 0.27)%** |
| PyTorch              | **(95.07 ± 0.31)%** |

The NumPy implementation successfully trains using manually derived and implemented backpropagation, without any ML libraries.

The same network was rebuilt in PyTorch as a reference. Both implementations agree within the variation caused by random initialisation and data shuffling.

## Optimisers (EMNIST)

The `optimisers/` folder trains a wider network on **EMNIST (balanced split) instead of MNIST**. The network is a 784 → 128 → 128 → 47 MLP. SGD, SGD with momentum and AdamW are implemented from scratch in `optimisers.py` and compared against their PyTorch equivalents.

Mean + standard deviation on the EMNIST test set over 10 runs. On each run, every optimiser starts from the same initial weights and sees the same shuffle order:

| Optimiser                 | Learning rate |        From scratch |         `torch.optim` |
| ------------------------- | ------------: | ------------------: | ------------------: |
| SGD                       |           0.1 | **(82.91 ± 0.30)%** | **(82.99 ± 0.29)%** |
| SGD + momentum (μ = 0.9)  |          0.01 | **(82.89 ± 0.34)%** | **(82.90 ± 0.30)%** |
| AdamW (weight decay 0.01) |         0.001 | **(83.33 ± 0.24)%** | **(83.34 ± 0.19)%** |

Each from-scratch optimiser matches its PyTorch counterpart within run-to-run variation.

SGD and SGD with momentum reach the same test accuracy. AdamW is about 0.4 percentage points higher at the learning rates tested.

## Project structure

```text
neuralnet-from-scratch/
├── README.md
├── .gitignore
├── mlp/                           # Multilayer perceptron from scratch
│   ├── guess.py                   # NumPy predictions and test accuracy
│   ├── training.py                # Train the NumPy network
│   ├── guess_with_pytorch.py      # PyTorch predictions and test accuracy
│   ├── training_with_pytorch.py   # Train the PyTorch network
├── optimisers/                    # Optimisers from scratch, trained on EMNIST
│   ├── model.py                   # MLP forward pass
│   ├── optimisers.py              # SGD, SGD + momentum, AdamW
│   ├── train.py                   # Training
│   ├── test.py                    # Predictions and test accuracy
│   └── compare_optimisers.py      
└── data/
    ├── MNIST/raw/
    │   ├── *-ubyte.gz             
    └── EMNIST/                    # Downloaded on first run
```

## Getting Started

### Prerequisites

- Python 3.8+

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/syn9139/neuralnet-from-scratch.git](https://github.com/syn9139/neuralnet-from-scratch.git)
   cd neuralnet-from-scratch
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy matplotlib
   ```
   or if using the script with PyTorch dependency,
      ```bash
   pip install numpy matplotlib torch torchvision
   ```

### Usage

1. **Train the network:**
   ```bash
   cd mlp
   python training.py
   ```
   *Trains the model from scratch and saves the parameters to `weights.npz`.*

2. **Run predictions:**
   ```bash
   python guess.py
   ```
   *Loads `weights.npz` and displays sample test images with predictions.*

## Acknowledgements

- [3Blue1Brown's Neural Network Series](https://www.3blue1brown.com/?topic=neural-networks) for the mathematics and visual intuition behind backpropagation.
- Yann LeCun and Corinna Cortes for the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).
