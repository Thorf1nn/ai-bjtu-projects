"""
Pooling layer implementations
"""
import numpy as np
from typing import Tuple, Union, Optional
from ..core.layer import Layer
from ..core.tensor import Tensor
from ..utils.conv_utils import maxpool2d_forward, maxpool2d_backward, avgpool2d_forward, avgpool2d_backward


class MaxPooling2D(Layer):
    """
    2D Max Pooling layer.
    """
    
    def __init__(self,
                 pool_size: Union[int, Tuple[int, int]] = 2,
                 stride: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: Union[int, Tuple[int, int]] = 0,
                 name: Optional[str] = None):
        """
        Initialize MaxPooling2D layer.
        
        Args:
            pool_size: Size of the pooling window
            stride: Stride of the pooling operation (defaults to pool_size)
            padding: Padding applied to input
            name: Layer name
        """
        super().__init__(name)
        
        # Handle pool_size as tuple
        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        else:
            self.pool_size = pool_size
            
        # Handle stride as tuple (default to pool_size)
        if stride is None:
            self.stride = self.pool_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
            
        # Handle padding as tuple
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
            
        # For backward pass
        self.last_input = None
        self.last_mask = None
        
    def build(self, input_shape):
        """
        Build the layer.
        
        Args:
            input_shape: Input shape (N, C, H, W)
        """
        super().build(input_shape)
        
        # Calculate output shape
        N, C, H, W = input_shape
        
        H_out = (H + 2 * self.padding[0] - self.pool_size[0]) // self.stride[0] + 1
        W_out = (W + 2 * self.padding[1] - self.pool_size[1]) // self.stride[1] + 1
        
        self.output_shape = (N, C, H_out, W_out)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor with shape (N, C, H, W)
            
        Returns:
            Output tensor
        """
        # Store input for backward pass
        self.last_input = x
        
        # Perform max pooling
        # For simplicity, assume square pooling and same padding for both dimensions
        output_data, mask = maxpool2d_forward(
            x.data, self.pool_size[0], self.pool_size[1], 
            self.stride[0], self.padding[0]
        )
        
        # Store mask for backward pass
        self.last_mask = mask
        
        # Create output tensor
        output = Tensor(output_data, requires_grad=x.requires_grad)
        
        return output
    
    def backward(self, grad_output: Tensor) -> Tensor:
        """
        Backward pass.
        
        Args:
            grad_output: Gradient w.r.t. output
            
        Returns:
            Gradient w.r.t. input
        """
        if self.last_input is None or self.last_mask is None:
            raise RuntimeError("Forward pass must be called before backward pass")
        
        # Get gradient w.r.t. input
        grad_input_data = maxpool2d_backward(
            grad_output.data, self.last_mask, self.last_input.shape,
            self.pool_size[0], self.pool_size[1], self.stride[0], self.padding[0]
        )
        
        # Return gradient w.r.t. input
        grad_input = Tensor(grad_input_data, requires_grad=self.last_input.requires_grad)
        return grad_input
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'pool_size': self.pool_size,
            'stride': self.stride,
            'padding': self.padding
        })
        return config
    
    def __repr__(self):
        return (f"MaxPooling2D(pool_size={self.pool_size}, "
                f"stride={self.stride}, padding={self.padding})")


class AvgPooling2D(Layer):
    """
    2D Average Pooling layer.
    """
    
    def __init__(self,
                 pool_size: Union[int, Tuple[int, int]] = 2,
                 stride: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: Union[int, Tuple[int, int]] = 0,
                 name: Optional[str] = None):
        """
        Initialize AvgPooling2D layer.
        
        Args:
            pool_size: Size of the pooling window
            stride: Stride of the pooling operation (defaults to pool_size)
            padding: Padding applied to input
            name: Layer name
        """
        super().__init__(name)
        
        # Handle pool_size as tuple
        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        else:
            self.pool_size = pool_size
            
        # Handle stride as tuple (default to pool_size)
        if stride is None:
            self.stride = self.pool_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
            
        # Handle padding as tuple
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
            
        # For backward pass
        self.last_input = None
        
    def build(self, input_shape):
        """
        Build the layer.
        
        Args:
            input_shape: Input shape (N, C, H, W)
        """
        super().build(input_shape)
        
        # Calculate output shape
        N, C, H, W = input_shape
        
        H_out = (H + 2 * self.padding[0] - self.pool_size[0]) // self.stride[0] + 1
        W_out = (W + 2 * self.padding[1] - self.pool_size[1]) // self.stride[1] + 1
        
        self.output_shape = (N, C, H_out, W_out)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor with shape (N, C, H, W)
            
        Returns:
            Output tensor
        """
        # Store input for backward pass
        self.last_input = x
        
        # Perform average pooling
        output_data = avgpool2d_forward(
            x.data, self.pool_size[0], self.pool_size[1], 
            self.stride[0], self.padding[0]
        )
        
        # Create output tensor
        output = Tensor(output_data, requires_grad=x.requires_grad)
        
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
        
        # Get gradient w.r.t. input
        grad_input_data = avgpool2d_backward(
            grad_output.data, self.last_input.shape,
            self.pool_size[0], self.pool_size[1], self.stride[0], self.padding[0]
        )
        
        # Return gradient w.r.t. input
        grad_input = Tensor(grad_input_data, requires_grad=self.last_input.requires_grad)
        return grad_input
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'pool_size': self.pool_size,
            'stride': self.stride,
            'padding': self.padding
        })
        return config
    
    def __repr__(self):
        return (f"AvgPooling2D(pool_size={self.pool_size}, "
                f"stride={self.stride}, padding={self.padding})")


class GlobalMaxPooling2D(Layer):
    """
    Global Max Pooling layer.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize GlobalMaxPooling2D layer.
        
        Args:
            name: Layer name
        """
        super().__init__(name)
        self.last_input = None
        self.last_mask = None
        
    def build(self, input_shape):
        """
        Build the layer.
        
        Args:
            input_shape: Input shape (N, C, H, W)
        """
        super().build(input_shape)
        
        # Output shape: (N, C)
        N, C, H, W = input_shape
        self.output_shape = (N, C)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor with shape (N, C, H, W)
            
        Returns:
            Output tensor with shape (N, C)
        """
        self.last_input = x
        
        # Global max pooling
        N, C, H, W = x.shape
        x_reshaped = x.data.reshape(N, C, -1)  # (N, C, H*W)
        
        # Find max values and their indices
        max_values = np.max(x_reshaped, axis=2)  # (N, C)
        max_indices = np.argmax(x_reshaped, axis=2)  # (N, C)
        
        # Create mask for backward pass
        mask = np.zeros_like(x.data, dtype=bool)
        for n in range(N):
            for c in range(C):
                flat_idx = max_indices[n, c]
                h_idx, w_idx = np.unravel_index(flat_idx, (H, W))
                mask[n, c, h_idx, w_idx] = True
        
        self.last_mask = mask
        
        output = Tensor(max_values, requires_grad=x.requires_grad)
        return output
    
    def backward(self, grad_output: Tensor) -> Tensor:
        """
        Backward pass.
        
        Args:
            grad_output: Gradient w.r.t. output with shape (N, C)
            
        Returns:
            Gradient w.r.t. input with shape (N, C, H, W)
        """
        if self.last_input is None or self.last_mask is None:
            raise RuntimeError("Forward pass must be called before backward pass")
        
        # Expand gradients back to input shape
        grad_input_data = np.zeros_like(self.last_input.data)
        N, C = grad_output.shape
        
        for n in range(N):
            for c in range(C):
                grad_input_data[n, c] = self.last_mask[n, c] * grad_output.data[n, c]
        
        grad_input = Tensor(grad_input_data, requires_grad=self.last_input.requires_grad)
        return grad_input
    
    def __repr__(self):
        return "GlobalMaxPooling2D()"


class GlobalAvgPooling2D(Layer):
    """
    Global Average Pooling layer.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize GlobalAvgPooling2D layer.
        
        Args:
            name: Layer name
        """
        super().__init__(name)
        self.last_input = None
        
    def build(self, input_shape):
        """
        Build the layer.
        
        Args:
            input_shape: Input shape (N, C, H, W)
        """
        super().build(input_shape)
        
        # Output shape: (N, C)
        N, C, H, W = input_shape
        self.output_shape = (N, C)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor with shape (N, C, H, W)
            
        Returns:
            Output tensor with shape (N, C)
        """
        self.last_input = x
        
        # Global average pooling
        output_data = np.mean(x.data, axis=(2, 3))  # Average over H and W dimensions
        
        output = Tensor(output_data, requires_grad=x.requires_grad)
        return output
    
    def backward(self, grad_output: Tensor) -> Tensor:
        """
        Backward pass.
        
        Args:
            grad_output: Gradient w.r.t. output with shape (N, C)
            
        Returns:
            Gradient w.r.t. input with shape (N, C, H, W)
        """
        if self.last_input is None:
            raise RuntimeError("Forward pass must be called before backward pass")
        
        N, C, H, W = self.last_input.shape
        
        # Distribute gradients evenly across spatial dimensions
        grad_input_data = np.zeros_like(self.last_input.data)
        for n in range(N):
            for c in range(C):
                grad_input_data[n, c] = grad_output.data[n, c] / (H * W)
        
        grad_input = Tensor(grad_input_data, requires_grad=self.last_input.requires_grad)
        return grad_input
    
    def __repr__(self):
        return "GlobalAvgPooling2D()" 