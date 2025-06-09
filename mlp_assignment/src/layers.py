import numpy as np
from src.initializers import get_initializer
from src.activations import get_activation

class Layer:
    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_dim, output_dim, activation='linear', initializer='he_normal'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.activation = get_activation(activation)
        self.initializer = get_initializer(initializer)
        
        self.weights = self.initializer((input_dim, output_dim))
        self.biases = np.zeros((1, output_dim))
        
        self.input = None
        self.z = None
        
        self.grad_weights = None
        self.grad_biases = None

    def forward(self, inputs):
        self.input = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        output = self.activation(self.z)
        return output

    def backward(self, grad_output):
        # Calculate gradient of loss w.r.t. pre-activation output (z)
        grad_z = grad_output * self.activation.derivative(self.z)
        
        # Calculate gradient of loss w.r.t. weights
        self.grad_weights = np.dot(self.input.T, grad_z)
        
        # Calculate gradient of loss w.r.t. biases
        self.grad_biases = np.sum(grad_z, axis=0, keepdims=True)
        
        # Calculate gradient of loss w.r.t. input (to be passed to the previous layer)
        grad_input = np.dot(grad_z, self.weights.T)
        
        return grad_input
