# MLP from Scratch

This project is an implementation of a Multi-Layer Perceptron (MLP) from scratch in Python, using NumPy for numerical operations. This was created as a solution to an assignment.

## Features

-   Flexible MLP architecture definition.
-   Multiple activation functions (ReLU, Sigmoid, Tanh, Softmax).
-   Support for both Classification and Regression tasks.
-   Various weight initialization techniques (Random, Xavier, He).
-   Optimizers: SGD, Momentum, RMSProp, Adam.
-   Regularization: L1, L2, and Elastic Net.
-   Loss tracking and visualization.
-   Evaluation metrics including accuracy and confusion matrix.

## Project Structure

```
mlp_assignment/
├── data/                 # Datasets
├── notebooks/            # Jupyter notebooks for demonstration
├── src/                  # Source code for the MLP
└── requirements.txt      # Project dependencies
```

## Usage

1.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Run the Jupyter notebooks in the `notebooks` directory to see the MLP in action on MNIST (classification) and Boston Housing (regression) datasets. 

