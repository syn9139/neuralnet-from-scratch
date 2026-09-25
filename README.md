# neuralnet-from-scratch

A lightweight handwritten digit recogniser built without any ML libraries like PyTorch or TensorFlow, loosely based on 3b1b's neural network series. Data trained on MNIST.

## Features

- Neural network built from scratch using only NumPy without ML libraries
- Trains on the full 60000 image MNIST dataset.
- Forward propagation, cost and back propagation derived manually
- Predictions visualised via matplotlib, showing a test digit and neural network prediction side by side.

## Results

Accuracy on the 10,000-image MNIST test set using the included weights:

| Implementation       |   Accuracy |
| -------------------- | ---------: |
| NumPy (from scratch) | **93.84%** |
| PyTorch              | **93.55%** |

The NumPy implementation successfully trains using manually derived and implemented backpropagation, without automatic differentiation or ML libraries.


## Project structure

```text
neuralnet-from-scratch/
├── README.md
├── guess.py                   # NumPy predictions and test accuracy
├── training.py                # Train the NumPy network
├── weights.npz                # Saved NumPy parameters
├── guess_with_pytorch.py      # PyTorch predictions and test accuracy
├── training_with_pytorch.py   # Train the PyTorch network
├── weights_with_pytorch.npz   # Saved PyTorch parameters
└── data/                      # MNIST dataset
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

### Usage

1. **Train the network:**
   ```bash
   python training.py
   ```
   *Trains the model from scratch and saves the parameters to `weights.npz`.*

2. **Run predictions:**
   ```bash
   python ocr.py
   ```
   *Loads `weights.npz` and displays sample test images with predictions.*

## Acknowledgements

- [3Blue1Brown's Neural Network Series](https://www.3blue1brown.com/?topic=neural-networks) for the mathematics and visual intuition behind backpropagation.
- Yann LeCun and Corinna Cortes for the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).