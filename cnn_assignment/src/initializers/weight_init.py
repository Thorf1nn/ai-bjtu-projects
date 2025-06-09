"""
Weight initialization functions for neural network parameters
"""
import numpy as np
from ..core.tensor import Tensor
from typing import Tuple


def xavier_uniform(shape: Tuple[int, ...], fan_in: int = None, fan_out: int = None) -> Tensor:
    """
    Xavier uniform initialization (Glorot uniform).
    
    Args:
        shape: Shape of the tensor to initialize
        fan_in: Number of input units
        fan_out: Number of output units
        
    Returns:
        Initialized tensor
    """
    if fan_in is None or fan_out is None:
        # Estimate fan_in and fan_out from shape
        if len(shape) == 2:  # Dense layer
            fan_in, fan_out = shape[0], shape[1]
        elif len(shape) == 4:  # Conv layer (out_channels, in_channels, height, width)
            fan_in = shape[1] * shape[2] * shape[3]  # in_channels * kernel_height * kernel_width
            fan_out = shape[0] * shape[2] * shape[3]  # out_channels * kernel_height * kernel_width
        else:
            fan_in = fan_out = int(np.sqrt(np.prod(shape)))
    
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    data = np.random.uniform(-limit, limit, shape).astype(np.float32)
    return Tensor(data, requires_grad=True)


def xavier_normal(shape: Tuple[int, ...], fan_in: int = None, fan_out: int = None) -> Tensor:
    """
    Xavier normal initialization (Glorot normal).
    
    Args:
        shape: Shape of the tensor to initialize
        fan_in: Number of input units
        fan_out: Number of output units
        
    Returns:
        Initialized tensor
    """
    if fan_in is None or fan_out is None:
        # Estimate fan_in and fan_out from shape
        if len(shape) == 2:  # Dense layer
            fan_in, fan_out = shape[0], shape[1]
        elif len(shape) == 4:  # Conv layer
            fan_in = shape[1] * shape[2] * shape[3]
            fan_out = shape[0] * shape[2] * shape[3]
        else:
            fan_in = fan_out = int(np.sqrt(np.prod(shape)))
    
    std = np.sqrt(2.0 / (fan_in + fan_out))
    data = np.random.normal(0, std, shape).astype(np.float32)
    return Tensor(data, requires_grad=True)


def he_uniform(shape: Tuple[int, ...], fan_in: int = None) -> Tensor:
    """
    He uniform initialization (good for ReLU activations).
    
    Args:
        shape: Shape of the tensor to initialize
        fan_in: Number of input units
        
    Returns:
        Initialized tensor
    """
    if fan_in is None:
        if len(shape) == 2:  # Dense layer
            fan_in = shape[0]
        elif len(shape) == 4:  # Conv layer
            fan_in = shape[1] * shape[2] * shape[3]
        else:
            fan_in = int(np.sqrt(np.prod(shape)))
    
    limit = np.sqrt(6.0 / fan_in)
    data = np.random.uniform(-limit, limit, shape).astype(np.float32)
    return Tensor(data, requires_grad=True)


def he_normal(shape: Tuple[int, ...], fan_in: int = None) -> Tensor:
    """
    He normal initialization (good for ReLU activations).
    
    Args:
        shape: Shape of the tensor to initialize
        fan_in: Number of input units
        
    Returns:
        Initialized tensor
    """
    if fan_in is None:
        if len(shape) == 2:  # Dense layer
            fan_in = shape[0]
        elif len(shape) == 4:  # Conv layer
            fan_in = shape[1] * shape[2] * shape[3]
        else:
            fan_in = int(np.sqrt(np.prod(shape)))
    
    std = np.sqrt(2.0 / fan_in)
    data = np.random.normal(0, std, shape).astype(np.float32)
    return Tensor(data, requires_grad=True)


def uniform(shape: Tuple[int, ...], low: float = -0.1, high: float = 0.1) -> Tensor:
    """
    Uniform initialization.
    
    Args:
        shape: Shape of the tensor to initialize
        low: Lower bound
        high: Upper bound
        
    Returns:
        Initialized tensor
    """
    data = np.random.uniform(low, high, shape).astype(np.float32)
    return Tensor(data, requires_grad=True)


def normal(shape: Tuple[int, ...], mean: float = 0.0, std: float = 0.1) -> Tensor:
    """
    Normal (Gaussian) initialization.
    
    Args:
        shape: Shape of the tensor to initialize
        mean: Mean of the distribution
        std: Standard deviation
        
    Returns:
        Initialized tensor
    """
    data = np.random.normal(mean, std, shape).astype(np.float32)
    return Tensor(data, requires_grad=True)


def zeros(shape: Tuple[int, ...]) -> Tensor:
    """
    Zero initialization.
    
    Args:
        shape: Shape of the tensor to initialize
        
    Returns:
        Initialized tensor filled with zeros
    """
    data = np.zeros(shape, dtype=np.float32)
    return Tensor(data, requires_grad=True)


def ones(shape: Tuple[int, ...]) -> Tensor:
    """
    Ones initialization.
    
    Args:
        shape: Shape of the tensor to initialize
        
    Returns:
        Initialized tensor filled with ones
    """
    data = np.ones(shape, dtype=np.float32)
    return Tensor(data, requires_grad=True)


def constant(shape: Tuple[int, ...], value: float) -> Tensor:
    """
    Constant initialization.
    
    Args:
        shape: Shape of the tensor to initialize
        value: Constant value to fill
        
    Returns:
        Initialized tensor filled with constant value
    """
    data = np.full(shape, value, dtype=np.float32)
    return Tensor(data, requires_grad=True)


def get_initializer(name: str):
    """
    Get initializer function by name.
    
    Args:
        name: Name of the initializer
        
    Returns:
        Initializer function
    """
    initializers = {
        'xavier_uniform': xavier_uniform,
        'xavier_normal': xavier_normal,
        'glorot_uniform': xavier_uniform,
        'glorot_normal': xavier_normal,
        'he_uniform': he_uniform,
        'he_normal': he_normal,
        'uniform': uniform,
        'normal': normal,
        'zeros': zeros,
        'ones': ones,
        'constant': constant
    }
    
    if name not in initializers:
        raise ValueError(f"Unknown initializer: {name}")
    
    return initializers[name] 