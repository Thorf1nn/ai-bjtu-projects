"""
2D Convolution layer implementation
"""
import numpy as np
from typing import Tuple, Union, Optional, Callable
from ..core.layer import Layer
from ..core.tensor import Tensor
from ..activations.functions import get_activation
from ..initializers.weight_init import get_initializer
from ..utils.conv_utils import conv2d_forward, conv2d_backward


class Conv2D(Layer):
    """
    2D Convolution layer.
    """
    
    def __init__(self, 
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 use_bias: bool = True,
                 weight_initializer: str = 'he_normal',
                 bias_initializer: str = 'zeros',
                 activation: Optional[str] = None,
                 name: Optional[str] = None):
        """
        Initialize Conv2D layer.
        
        Args:
            out_channels: Number of output channels
            kernel_size: Size of the convolution kernel
            stride: Stride of the convolution
            padding: Padding applied to input
            dilation: Dilation rate of the convolution
            use_bias: Whether to use bias
            weight_initializer: Weight initialization method
            bias_initializer: Bias initialization method
            activation: Activation function name
            name: Layer name
        """
        super().__init__(name)
        
        self.out_channels = out_channels
        
        # Handle kernel_size as tuple
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        # Handle stride as tuple
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
            
        # Handle padding as tuple
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
            
        # Handle dilation as tuple
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation
            
        self.use_bias = use_bias
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        
        # Activation function
        self.activation_fn = None
        if activation is not None:
            self.activation_fn = get_activation(activation)
        
        # These will be set during build
        self.in_channels = None
        self.weight = None
        self.bias = None
        
        # For backward pass
        self.last_input = None
        
    def build(self, input_shape):
        """
        Build the layer.
        
        Args:
            input_shape: Input shape (N, C, H, W)
        """
        super().build(input_shape)
        
        # Extract input channels
        self.in_channels = input_shape[1]
        
        # Initialize weights
        weight_shape = (self.out_channels, self.in_channels, 
                       self.kernel_size[0], self.kernel_size[1])
        
        weight_init_fn = get_initializer(self.weight_initializer)
        self.weight = weight_init_fn(weight_shape)
        self._parameters['weight'] = self.weight
        
        # Initialize bias
        if self.use_bias:
            bias_init_fn = get_initializer(self.bias_initializer)
            self.bias = bias_init_fn((self.out_channels,))
            self._parameters['bias'] = self.bias
        
        # Calculate output shape
        N, C, H, W = input_shape
        
        # Account for dilation
        effective_kernel_h = self.kernel_size[0] + (self.kernel_size[0] - 1) * (self.dilation[0] - 1)
        effective_kernel_w = self.kernel_size[1] + (self.kernel_size[1] - 1) * (self.dilation[1] - 1)
        
        H_out = (H + 2 * self.padding[0] - effective_kernel_h) // self.stride[0] + 1
        W_out = (W + 2 * self.padding[1] - effective_kernel_w) // self.stride[1] + 1
        
        self.output_shape = (N, self.out_channels, H_out, W_out)
        
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
        
        # Perform convolution
        bias_data = self.bias.data if self.use_bias else None
        
        # Handle dilation by dilating the kernel
        if self.dilation != (1, 1):
            # For simplicity, we'll use a basic dilation implementation
            # In practice, you might want to optimize this further
            dilated_weight = self._dilate_kernel(self.weight.data)
            output_data = conv2d_forward(x.data, dilated_weight, bias_data, 
                                       self.stride, self.padding)
        else:
            output_data = conv2d_forward(x.data, self.weight.data, bias_data,
                                       self.stride, self.padding)
        
        # Create output tensor
        output = Tensor(output_data, requires_grad=x.requires_grad or self.weight.requires_grad)
        
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
        
        # Get gradients
        if self.dilation != (1, 1):
            dilated_weight = self._dilate_kernel(self.weight.data)
            grad_input_data, grad_weight_data, grad_bias_data = conv2d_backward(
                grad_output.data, self.last_input.data, dilated_weight,
                self.stride, self.padding
            )
            # Un-dilate the weight gradient
            grad_weight_data = self._undilate_kernel_grad(grad_weight_data)
        else:
            grad_input_data, grad_weight_data, grad_bias_data = conv2d_backward(
                grad_output.data, self.last_input.data, self.weight.data,
                self.stride, self.padding
            )
        
        # Update parameter gradients
        if self.weight.requires_grad:
            if self.weight.grad is None:
                self.weight.grad = np.zeros_like(self.weight.data)
            self.weight.grad += grad_weight_data
            
        if self.use_bias and self.bias.requires_grad:
            if self.bias.grad is None:
                self.bias.grad = np.zeros_like(self.bias.data)
            self.bias.grad += grad_bias_data
        
        # Return gradient w.r.t. input
        grad_input = Tensor(grad_input_data, requires_grad=self.last_input.requires_grad)
        return grad_input
    
    def _dilate_kernel(self, kernel: np.ndarray) -> np.ndarray:
        """
        Dilate kernel for dilated convolution.
        
        Args:
            kernel: Original kernel
            
        Returns:
            Dilated kernel
        """
        if self.dilation == (1, 1):
            return kernel
            
        C_out, C_in, K_h, K_w = kernel.shape
        dil_h, dil_w = self.dilation
        
        # Calculate dilated kernel size
        dilated_h = K_h + (K_h - 1) * (dil_h - 1)
        dilated_w = K_w + (K_w - 1) * (dil_w - 1)
        
        # Create dilated kernel
        dilated_kernel = np.zeros((C_out, C_in, dilated_h, dilated_w), dtype=kernel.dtype)
        
        for i in range(K_h):
            for j in range(K_w):
                dilated_kernel[:, :, i * dil_h, j * dil_w] = kernel[:, :, i, j]
                
        return dilated_kernel
    
    def _undilate_kernel_grad(self, dilated_grad: np.ndarray) -> np.ndarray:
        """
        Extract original kernel gradient from dilated gradient.
        
        Args:
            dilated_grad: Gradient w.r.t. dilated kernel
            
        Returns:
            Gradient w.r.t. original kernel
        """
        if self.dilation == (1, 1):
            return dilated_grad
            
        dil_h, dil_w = self.dilation
        original_shape = self.weight.shape
        grad = np.zeros(original_shape, dtype=dilated_grad.dtype)
        
        K_h, K_w = original_shape[2], original_shape[3]
        
        for i in range(K_h):
            for j in range(K_w):
                grad[:, :, i, j] = dilated_grad[:, :, i * dil_h, j * dil_w]
                
        return grad
        
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'dilation': self.dilation,
            'use_bias': self.use_bias,
            'weight_initializer': self.weight_initializer,
            'bias_initializer': self.bias_initializer
        })
        return config
    
    def __repr__(self):
        return (f"Conv2D(out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding}, use_bias={self.use_bias})") 