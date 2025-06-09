"""
Optimized convolution utilities using im2col/col2im
"""
import numpy as np
from typing import Tuple, Union


def im2col(input_data: np.ndarray, kernel_h: int, kernel_w: int, 
           stride: int = 1, pad: int = 0) -> np.ndarray:
    """
    Convert image patches to columns for efficient convolution.
    
    Args:
        input_data: Input data with shape (N, C, H, W)
        kernel_h: Kernel height
        kernel_w: Kernel width
        stride: Stride for convolution
        pad: Padding size
        
    Returns:
        Column matrix with shape (C*kernel_h*kernel_w, N*out_h*out_w)
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - kernel_h) // stride + 1
    out_w = (W + 2 * pad - kernel_w) // stride + 1
    
    # Apply padding
    img = np.pad(input_data, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    
    # Create column matrix
    col = np.ndarray((N, C, kernel_h, kernel_w, out_h, out_w), dtype=input_data.dtype)
    
    for j in range(kernel_h):
        j_max = j + stride * out_h
        for i in range(kernel_w):
            i_max = i + stride * out_w
            col[:, :, j, i, :, :] = img[:, :, j:j_max:stride, i:i_max:stride]
    
    col = col.transpose(1, 2, 3, 0, 4, 5).reshape(C * kernel_h * kernel_w, -1)
    return col


def col2im(col: np.ndarray, input_shape: Tuple[int, int, int, int], 
           kernel_h: int, kernel_w: int, stride: int = 1, pad: int = 0) -> np.ndarray:
    """
    Convert columns back to image format.
    
    Args:
        col: Column matrix with shape (C*kernel_h*kernel_w, N*out_h*out_w)
        input_shape: Original input shape (N, C, H, W)
        kernel_h: Kernel height
        kernel_w: Kernel width
        stride: Stride for convolution
        pad: Padding size
        
    Returns:
        Image data with shape (N, C, H, W)
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - kernel_h) // stride + 1
    out_w = (W + 2 * pad - kernel_w) // stride + 1
    
    col = col.reshape(C, kernel_h, kernel_w, N, out_h, out_w).transpose(3, 0, 1, 2, 4, 5)
    
    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1), dtype=col.dtype)
    
    for j in range(kernel_h):
        j_max = j + stride * out_h
        for i in range(kernel_w):
            i_max = i + stride * out_w
            img[:, :, j:j_max:stride, i:i_max:stride] += col[:, :, j, i, :, :]
    
    return img[:, :, pad:H + pad, pad:W + pad]


def conv2d_forward(input_data: np.ndarray, weight: np.ndarray, bias: np.ndarray = None,
                   stride: Union[int, Tuple[int, int]] = 1, 
                   padding: Union[int, Tuple[int, int]] = 0) -> np.ndarray:
    """
    Forward pass for 2D convolution using im2col.
    
    Args:
        input_data: Input with shape (N, C_in, H, W)
        weight: Weight with shape (C_out, C_in, kernel_h, kernel_w)
        bias: Bias with shape (C_out,) or None
        stride: Stride for convolution
        padding: Padding for convolution
        
    Returns:
        Output with shape (N, C_out, H_out, W_out)
    """
    # Handle stride and padding as tuples
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
        
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding
    
    N, C_in, H, W = input_data.shape
    C_out, C_in, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    H_out = (H + 2 * pad_h - kernel_h) // stride_h + 1
    W_out = (W + 2 * pad_w - kernel_w) // stride_w + 1
    
    # Convert input to column matrix
    # For simplicity, assume stride_h == stride_w and pad_h == pad_w
    col = im2col(input_data, kernel_h, kernel_w, stride_h, pad_h)
    
    # Reshape weight for matrix multiplication
    weight_col = weight.reshape(C_out, -1)
    
    # Perform convolution as matrix multiplication
    out = np.dot(weight_col, col)
    
    # Add bias if provided
    if bias is not None:
        out += bias.reshape(-1, 1)
    
    # Reshape output
    out = out.reshape(C_out, N, H_out, W_out).transpose(1, 0, 2, 3)
    
    return out


def conv2d_backward(grad_output: np.ndarray, input_data: np.ndarray, weight: np.ndarray,
                    stride: Union[int, Tuple[int, int]] = 1,
                    padding: Union[int, Tuple[int, int]] = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Backward pass for 2D convolution.
    
    Args:
        grad_output: Gradient w.r.t. output with shape (N, C_out, H_out, W_out)
        input_data: Input with shape (N, C_in, H, W)
        weight: Weight with shape (C_out, C_in, kernel_h, kernel_w)
        stride: Stride for convolution
        padding: Padding for convolution
        
    Returns:
        Tuple of (grad_input, grad_weight, grad_bias)
    """
    # Handle stride and padding as tuples
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
        
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding
    
    N, C_in, H, W = input_data.shape
    C_out, C_in, kernel_h, kernel_w = weight.shape
    N, C_out, H_out, W_out = grad_output.shape
    
    # Gradient w.r.t. bias
    grad_bias = grad_output.sum(axis=(0, 2, 3))
    
    # Reshape grad_output for matrix operations
    grad_output_col = grad_output.transpose(1, 0, 2, 3).reshape(C_out, -1)
    
    # Convert input to column matrix
    input_col = im2col(input_data, kernel_h, kernel_w, stride_h, pad_h)
    
    # Gradient w.r.t. weight
    grad_weight = np.dot(grad_output_col, input_col.T).reshape(weight.shape)
    
    # Gradient w.r.t. input
    weight_col = weight.reshape(C_out, -1)
    grad_input_col = np.dot(weight_col.T, grad_output_col)
    grad_input = col2im(grad_input_col, input_data.shape, kernel_h, kernel_w, stride_h, pad_h)
    
    return grad_input, grad_weight, grad_bias


def maxpool2d_forward(input_data: np.ndarray, pool_h: int, pool_w: int,
                      stride: int = None, padding: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward pass for 2D max pooling.
    
    Args:
        input_data: Input with shape (N, C, H, W)
        pool_h: Pool height
        pool_w: Pool width
        stride: Stride (defaults to pool size)
        padding: Padding
        
    Returns:
        Tuple of (output, mask) where mask is for backward pass
    """
    if stride is None:
        stride = pool_h
        
    N, C, H, W = input_data.shape
    H_out = (H + 2 * padding - pool_h) // stride + 1
    W_out = (W + 2 * padding - pool_w) // stride + 1
    
    # Apply padding
    if padding > 0:
        input_padded = np.pad(input_data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
                             mode='constant', constant_values=-np.inf)
    else:
        input_padded = input_data
    
    # Create output and mask arrays
    output = np.zeros((N, C, H_out, W_out), dtype=input_data.dtype)
    mask = np.zeros_like(input_padded, dtype=bool)
    
    for h in range(H_out):
        for w in range(W_out):
            h_start = h * stride
            h_end = h_start + pool_h
            w_start = w * stride
            w_end = w_start + pool_w
            
            # Get the window
            window = input_padded[:, :, h_start:h_end, w_start:w_end]
            
            # Find max values and their positions
            max_vals = np.max(window.reshape(N, C, -1), axis=2)
            output[:, :, h, w] = max_vals
            
            # Create mask for backward pass
            window_flat = window.reshape(N, C, -1)
            max_indices = np.argmax(window_flat, axis=2)
            
            for n in range(N):
                for c in range(C):
                    max_idx = max_indices[n, c]
                    h_idx, w_idx = np.unravel_index(max_idx, (pool_h, pool_w))
                    mask[n, c, h_start + h_idx, w_start + w_idx] = True
    
    return output, mask


def maxpool2d_backward(grad_output: np.ndarray, mask: np.ndarray, input_shape: Tuple[int, int, int, int],
                       pool_h: int, pool_w: int, stride: int = None, padding: int = 0) -> np.ndarray:
    """
    Backward pass for 2D max pooling.
    
    Args:
        grad_output: Gradient w.r.t. output with shape (N, C, H_out, W_out)
        mask: Mask from forward pass
        input_shape: Original input shape
        pool_h: Pool height
        pool_w: Pool width
        stride: Stride
        padding: Padding
        
    Returns:
        Gradient w.r.t. input
    """
    if stride is None:
        stride = pool_h
        
    N, C, H_out, W_out = grad_output.shape
    grad_input = np.zeros(mask.shape, dtype=grad_output.dtype)
    
    for h in range(H_out):
        for w in range(W_out):
            h_start = h * stride
            h_end = h_start + pool_h
            w_start = w * stride
            w_end = w_start + pool_w
            
            # Distribute gradients to the max positions
            window_mask = mask[:, :, h_start:h_end, w_start:w_end]
            grad_input[:, :, h_start:h_end, w_start:w_end] += (
                window_mask * grad_output[:, :, h:h+1, w:w+1]
            )
    
    # Remove padding if it was applied
    if padding > 0:
        grad_input = grad_input[:, :, padding:-padding, padding:-padding]
    
    return grad_input


def avgpool2d_forward(input_data: np.ndarray, pool_h: int, pool_w: int,
                      stride: int = None, padding: int = 0) -> np.ndarray:
    """
    Forward pass for 2D average pooling.
    
    Args:
        input_data: Input with shape (N, C, H, W)
        pool_h: Pool height
        pool_w: Pool width
        stride: Stride (defaults to pool size)
        padding: Padding
        
    Returns:
        Output with shape (N, C, H_out, W_out)
    """
    if stride is None:
        stride = pool_h
        
    N, C, H, W = input_data.shape
    H_out = (H + 2 * padding - pool_h) // stride + 1
    W_out = (W + 2 * padding - pool_w) // stride + 1
    
    # Apply padding
    if padding > 0:
        input_padded = np.pad(input_data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
                             mode='constant', constant_values=0)
    else:
        input_padded = input_data
    
    # Create output array
    output = np.zeros((N, C, H_out, W_out), dtype=input_data.dtype)
    
    for h in range(H_out):
        for w in range(W_out):
            h_start = h * stride
            h_end = h_start + pool_h
            w_start = w * stride
            w_end = w_start + pool_w
            
            # Average over the window
            window = input_padded[:, :, h_start:h_end, w_start:w_end]
            output[:, :, h, w] = np.mean(window, axis=(2, 3))
    
    return output


def avgpool2d_backward(grad_output: np.ndarray, input_shape: Tuple[int, int, int, int],
                       pool_h: int, pool_w: int, stride: int = None, padding: int = 0) -> np.ndarray:
    """
    Backward pass for 2D average pooling.
    
    Args:
        grad_output: Gradient w.r.t. output with shape (N, C, H_out, W_out)
        input_shape: Original input shape
        pool_h: Pool height
        pool_w: Pool width
        stride: Stride
        padding: Padding
        
    Returns:
        Gradient w.r.t. input
    """
    if stride is None:
        stride = pool_h
        
    N, C, H, W = input_shape
    N, C, H_out, W_out = grad_output.shape
    
    # Account for padding in gradient shape
    if padding > 0:
        grad_input = np.zeros((N, C, H + 2 * padding, W + 2 * padding), dtype=grad_output.dtype)
    else:
        grad_input = np.zeros(input_shape, dtype=grad_output.dtype)
    
    for h in range(H_out):
        for w in range(W_out):
            h_start = h * stride
            h_end = h_start + pool_h
            w_start = w * stride
            w_end = w_start + pool_w
            
            # Distribute gradients evenly across the pool window
            grad_input[:, :, h_start:h_end, w_start:w_end] += (
                grad_output[:, :, h:h+1, w:w+1] / (pool_h * pool_w)
            )
    
    # Remove padding if it was applied
    if padding > 0:
        grad_input = grad_input[:, :, padding:-padding, padding:-padding]
    
    return grad_input 