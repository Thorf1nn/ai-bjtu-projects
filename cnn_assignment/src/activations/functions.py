"""
Activation functions implemented as classes for serialization
"""
import numpy as np
from ..core.tensor import Tensor, _empty_backward

class Activation:
    """Base class for activation functions."""
    
    def __call__(self, x: Tensor) -> Tensor:
        """
        Forward pass for the activation function.
        This also sets up the backward pass.
        """
        result = Tensor(self.forward(x.data), requires_grad=x.requires_grad)
        
        if x.requires_grad:
            result._prev = {x}
            result.context = x # Store input tensor in context
            result._backward = self._perform_backward
            
        return result

    def _perform_backward(self, output_tensor: Tensor):
        """
        Performs the backward pass for the activation function.
        This method is pickleable.
        
        Args:
            output_tensor: The tensor resulting from the forward pass.
        """
        input_tensor = output_tensor.context
        if input_tensor.requires_grad:
            # Derivative of the activation function w.r.t its input
            activation_grad = self.backward(input_tensor.data)
            
            # Chain rule: gradient to propagate is downstream_grad * activation_grad
            grad_to_propagate = output_tensor.grad * activation_grad
            
            # Accumulate gradient in the input tensor
            if input_tensor.grad is None:
                input_tensor.grad = np.zeros_like(input_tensor.data)
            input_tensor.grad += grad_to_propagate

    def forward(self, x_data: np.ndarray) -> np.ndarray:
        """The forward computation."""
        raise NotImplementedError

    def backward(self, x_data: np.ndarray) -> np.ndarray:
        """The backward computation (the derivative of the activation)."""
        raise NotImplementedError

class ReLU(Activation):
    """ReLU activation function."""
    def forward(self, x_data: np.ndarray) -> np.ndarray:
        return np.maximum(0, x_data)
    
    def backward(self, x_data: np.ndarray) -> np.ndarray:
        return (x_data > 0).astype(np.float32)

class LeakyReLU(Activation):
    """Leaky ReLU activation function."""
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x_data: np.ndarray) -> np.ndarray:
        return np.where(x_data > 0, x_data, self.negative_slope * x_data)

    def backward(self, x_data: np.ndarray) -> np.ndarray:
        return np.where(x_data > 0, 1.0, self.negative_slope)

class Sigmoid(Activation):
    """Sigmoid activation function."""
    def forward(self, x_data: np.ndarray) -> np.ndarray:
        clipped_data = np.clip(x_data, -500, 500)
        self.output_data = 1.0 / (1.0 + np.exp(-clipped_data))
        return self.output_data

    def backward(self, x_data: np.ndarray) -> np.ndarray:
        return self.output_data * (1.0 - self.output_data)

class Tanh(Activation):
    """Hyperbolic tangent activation function."""
    def forward(self, x_data: np.ndarray) -> np.ndarray:
        self.output_data = np.tanh(x_data)
        return self.output_data

    def backward(self, x_data: np.ndarray) -> np.ndarray:
        return 1.0 - self.output_data ** 2

class Softmax(Activation):
    """Softmax activation function."""
    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def forward(self, x_data: np.ndarray) -> np.ndarray:
        x_max = np.max(x_data, axis=self.axis, keepdims=True)
        exp_data = np.exp(x_data - x_max)
        sum_exp = np.sum(exp_data, axis=self.axis, keepdims=True)
        self.output_data = exp_data / sum_exp
        return self.output_data

    def backward(self, x_data: np.ndarray) -> np.ndarray:
        # This is a simplified gradient for when used with CrossEntropyLoss.
        # The true gradient is complex, but this works because the gradient
        # of (Softmax + CrossEntropy) is very simple (predictions - targets),
        # and that calculation is handled in the loss function.
        return np.ones_like(x_data)

class Linear(Activation):
    """Linear (identity) activation function."""
    def forward(self, x_data: np.ndarray) -> np.ndarray:
        return x_data
        
    def backward(self, x_data: np.ndarray) -> np.ndarray:
        return np.ones_like(x_data)


def get_activation(name: str):
    """
    Get activation function instance by name.
    
    Args:
        name: Name of the activation function
        
    Returns:
        Activation function instance
    """
    activations = {
        'relu': ReLU,
        'leaky_relu': LeakyReLU,
        'sigmoid': Sigmoid,
        'tanh': Tanh,
        'softmax': Softmax,
        'linear': Linear,
        None: Linear,
        'none': Linear
    }
    
    if name not in activations:
        raise ValueError(f"Unknown activation function: {name}")
    
    # Return an instance of the class
    return activations[name]() 