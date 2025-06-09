"""
Flatten layer implementation
"""
import numpy as np
from typing import Optional
from ..core.layer import Layer
from ..core.tensor import Tensor


class Flatten(Layer):
    """
    Flatten layer to convert multi-dimensional input to 2D.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize Flatten layer.
        
        Args:
            name: Layer name
        """
        super().__init__(name)
        self.last_input_shape = None
        
    def build(self, input_shape):
        """
        Build the layer.
        
        Args:
            input_shape: Input shape (N, C, H, W) or any multi-dimensional shape
        """
        super().build(input_shape)
        
        # Calculate output shape
        # Flatten all dimensions except the first (batch dimension)
        N = input_shape[0]
        flattened_size = np.prod(input_shape[1:])
        self.output_shape = (N, flattened_size)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor with shape (N, ...)
            
        Returns:
            Output tensor with shape (N, flattened_size)
        """
        # Store input shape for backward pass
        self.last_input_shape = x.shape
        
        # Flatten the tensor
        batch_size = x.shape[0]
        flattened_size = np.prod(x.shape[1:])
        
        output_data = x.data.reshape(batch_size, flattened_size)
        output = Tensor(output_data, requires_grad=x.requires_grad)
        
        return output
    
    def backward(self, grad_output: Tensor) -> Tensor:
        """
        Backward pass.
        
        Args:
            grad_output: Gradient w.r.t. output with shape (N, flattened_size)
            
        Returns:
            Gradient w.r.t. input with original input shape
        """
        if self.last_input_shape is None:
            raise RuntimeError("Forward pass must be called before backward pass")
        
        # Reshape gradient back to original input shape
        grad_input_data = grad_output.data.reshape(self.last_input_shape)
        grad_input = Tensor(grad_input_data, requires_grad=grad_output.requires_grad)
        
        return grad_input
    
    def __repr__(self):
        return "Flatten()" 