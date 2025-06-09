"""
Dense (Fully Connected) layer implementation
"""
import numpy as np
from typing import Optional
from ..core.layer import Layer
from ..core.tensor import Tensor
from ..activations.functions import get_activation
from ..initializers.weight_init import get_initializer


class Dense(Layer):
    """
    Fully connected (dense) layer.
    """
    
    def __init__(self,
                 units: int,
                 use_bias: bool = True,
                 weight_initializer: str = 'xavier_normal',
                 bias_initializer: str = 'zeros',
                 activation: Optional[str] = None,
                 name: Optional[str] = None):
        """
        Initialize Dense layer.
        
        Args:
            units: Number of output units
            use_bias: Whether to use bias
            weight_initializer: Weight initialization method
            bias_initializer: Bias initialization method
            activation: Activation function name
            name: Layer name
        """
        super().__init__(name)
        
        self.units = units
        self.use_bias = use_bias
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        
        # Activation function
        self.activation_fn = None
        if activation is not None:
            self.activation_fn = get_activation(activation)
        
        # These will be set during build
        self.input_units = None
        self.weight = None
        self.bias = None
        
        # For backward pass
        self.last_input = None
        
    def build(self, input_shape):
        """
        Build the layer.
        
        Args:
            input_shape: Input shape (..., input_units)
        """
        super().build(input_shape)
        
        # Extract input units (last dimension)
        self.input_units = input_shape[-1]
        
        # Initialize weights
        weight_shape = (self.input_units, self.units)
        weight_init_fn = get_initializer(self.weight_initializer)
        self.weight = weight_init_fn(weight_shape)
        self._parameters['weight'] = self.weight
        
        # Initialize bias
        if self.use_bias:
            bias_init_fn = get_initializer(self.bias_initializer)
            self.bias = bias_init_fn((self.units,))
            self._parameters['bias'] = self.bias
        
        # Calculate output shape
        self.output_shape = input_shape[:-1] + (self.units,)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor with shape (..., input_units)
            
        Returns:
            Output tensor with shape (..., units)
        """
        # Store input for backward pass
        self.last_input = x
        
        # Perform matrix multiplication
        output = x @ self.weight
        
        # Add bias if used
        if self.use_bias:
            output = output + self.bias
        
        # Apply activation if specified
        if self.activation_fn is not None:
            output = self.activation_fn(output)
            
        return output
    
    def backward(self, grad_output: Tensor) -> Tensor:
        """
        Backward pass.
        
        Args:
            grad_output: Gradient w.r.t. output
            
        Returns:
            Gradient w.r.t. input
        """
        if self.last_input is None:
            raise RuntimeError("Forward pass must be called before backward pass")
        
        # Gradient w.r.t. weight: input^T @ grad_output
        grad_weight_data = np.dot(self.last_input.data.T, grad_output.data)
        
        # Update weight gradient
        if self.weight.requires_grad:
            if self.weight.grad is None:
                self.weight.grad = np.zeros_like(self.weight.data)
            self.weight.grad += grad_weight_data
            
        # Gradient w.r.t. bias: sum over batch dimension
        if self.use_bias and self.bias.requires_grad:
            grad_bias_data = np.sum(grad_output.data, axis=0)
            if self.bias.grad is None:
                self.bias.grad = np.zeros_like(self.bias.data)
            self.bias.grad += grad_bias_data
        
        # Gradient w.r.t. input: grad_output @ weight^T
        grad_input_data = np.dot(grad_output.data, self.weight.data.T)
        grad_input = Tensor(grad_input_data, requires_grad=self.last_input.requires_grad)
        
        return grad_input
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'units': self.units,
            'use_bias': self.use_bias,
            'weight_initializer': self.weight_initializer,
            'bias_initializer': self.bias_initializer
        })
        return config
    
    def __repr__(self):
        return f"Dense(units={self.units}, use_bias={self.use_bias})" 