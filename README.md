# neuralnet-from-scratch

A lightweight handwritten digit recogniser built without any ML libraries like PyTorch or TensorFlow, loosely based on 3b1b's neural network series. Data trained on MNIST.

## Features

- Neural network built from scratch using only NumPy without ML libraries.
- Trains on the full 60000 image MNIST dataset.
- Forward pass, loss and back propagation derived manually
- 784 --> 16 --> 16 --> 10 MLP with ReLU hidden layers.
- Softmax output layer with cross-entropy loss.
- Predictions visualised via matplotlib, showing a test digit and neural network prediction side by side.
- The same neural network but with PyTorch for reference.

## Results

Mean + standard deviation on the 10,000-image MNIST test set using 10 random seeds:

| Implementation       |   Accuracy |
| -------------------- | ---------: |
| NumPy (from scratch) | **(95.00 ± 0.27)%** |
| PyTorch              | **(95.07 ± 0.31)%** |

The NumPy implementation successfully trains using manually derived and implemented backpropagation, without any ML libraries.

The same network was rebuilt in PyTorch as a reference. Both implementations agree within the variation caused by random initialisation and data shuffling.


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
└── data/
    └── MNIST/raw/
        ├── *-ubyte.gz             # Compressed MNIST dataset
```

Weights files are not tracked, so run the training scripts before the `guess` scripts.

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
