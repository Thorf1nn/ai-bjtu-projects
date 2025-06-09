# CNN Implementation from Scratch

## Overview
This project implements a complete Convolutional Neural Network (CNN) framework from scratch without using existing deep learning frameworks like PyTorch or TensorFlow. The implementation covers all core components needed for CNN training and inference.

## Features

### Core Components
1. **Flexible CNN Architecture Definition**
2. **Multiple Activation Functions** (ReLU, LeakyReLU, Sigmoid, Tanh)
3. **Classification and Regression Support**
4. **Weight Initialization Options** (Xavier, He, Random)
5. **Multiple Optimizers** (SGD, Momentum, RMSprop, Adam)
6. **Regularization** (L1, L2, Elastic Net)
7. **Stop Criteria** for training

### Layer Implementations
- Conv2D (with optimized im2col/col2im)
- MaxPooling2D / AvgPooling2D
- Dropout
- BatchNormalization
- Flatten
- Fully Connected (Dense)

### Advanced Architectures
- Inception Module
- Residual Block
- Depthwise Convolution
- Bottleneck Block

### Optimized Operations
- im2col/col2im for efficient convolution
- FFT-based convolution (optional)
- Parallel processing support

## Project Structure
```
cnn_assignment/
├── src/
│   ├── core/                 # Core framework components
│   ├── layers/              # Layer implementations
│   ├── optimizers/          # Optimizer implementations
│   ├── activations/         # Activation functions
│   ├── initializers/        # Weight initialization
│   ├── regularizers/        # Regularization methods
│   ├── models/              # Pre-built architectures
│   └── utils/               # Utility functions
├── examples/                # Usage examples
├── tests/                   # Unit tests
├── data/                    # Dataset utilities
└── experiments/             # Training experiments

```

## Usage Example
```python
from src.models import CNN
from src.layers import Conv2D, MaxPooling2D, Dense
from src.optimizers import Adam

# Define model architecture
model = CNN()
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile model
model.compile(optimizer=Adam(), loss='categorical_crossentropy')

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32)
```

## Supported Datasets
- CIFAR-10/100
- MNIST
- FashionMNIST
- Custom datasets

## Requirements
- NumPy
- Matplotlib (for visualization)
- Pillow (for image processing)
- Optional: Numba (for JIT compilation) 