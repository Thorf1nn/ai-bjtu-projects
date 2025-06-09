"""
Tensor class with automatic differentiation support
"""
import numpy as np
from typing import Optional, Tuple, Union, List, Callable


def _empty_backward():
    """A placeholder for the backward function for tensors that don't have one."""
    pass


class Tensor:
    """
    A tensor class that supports automatic differentiation.
    This is the core data structure for our CNN framework.
    """
    
    def __init__(self, data, requires_grad: bool = False, 
                 grad_fn: Optional[Callable] = None):
        """
        Initialize a tensor.
        
        Args:
            data: The actual data as numpy array
            requires_grad: Whether this tensor requires gradient computation
            grad_fn: Function to compute gradients during backpropagation
        """
        if isinstance(data, (int, float)):
            data = np.array(data)
        elif isinstance(data, list):
            data = np.array(data)
        
        self.data = data.astype(np.float32)
        self.requires_grad = requires_grad
        self.grad_fn = None  # Deprecated, use _backward and _prev
        self._backward = _empty_backward
        self._prev = set()
        self.grad = None
        self.context = None # For storing information needed for backward pass
        self._children = []  # For tracking computation graph
        
        if requires_grad:
            self.grad = np.zeros_like(self.data)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the tensor."""
        return self.data.shape
    
    @property 
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return self.data.ndim
    
    @property
    def size(self) -> int:
        """Return the total number of elements."""
        return self.data.size
    
    @property
    def dtype(self):
        """Return the data type."""
        return self.data.dtype
    
    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
    
    def __str__(self) -> str:
        return f"Tensor({self.data})"
    
    def __getitem__(self, key):
        """Support indexing."""
        return Tensor(self.data[key], requires_grad=self.requires_grad)
    
    def __setitem__(self, key, value):
        """Support item assignment."""
        if isinstance(value, Tensor):
            self.data[key] = value.data
        else:
            self.data[key] = value
    
    # Arithmetic operations
    def __add__(self, other):
        return self.add(other)
    
    def __radd__(self, other):
        return self.add(other)
    
    def __sub__(self, other):
        return self.sub(other)
    
    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(np.array(other))
        return other.sub(self)
    
    def __mul__(self, other):
        return self.mul(other)
    
    def __rmul__(self, other):
        return self.mul(other)
    
    def __truediv__(self, other):
        return self.div(other)
    
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(np.array(other))
        return other.div(self)
    
    def __matmul__(self, other):
        return self.matmul(other)
    
    def __pow__(self, other):
        return self.pow(other)
    
    def __neg__(self):
        return self.mul(-1)
    
    # Core operations with gradient support
    def add(self, other):
        """Addition operation."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        result = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        
        if result.requires_grad:
            result._prev = {self, other}
            result.context = (self, other) # Store the actual tensors, not just shapes
            result._backward = self._backward_add
            
        return result

    def _backward_add(self, output_tensor: 'Tensor'):
        input_tensor, other_tensor = output_tensor.context
        grad_output = output_tensor.grad
        
        if input_tensor.requires_grad:
            grad = grad_output
            # Handle broadcasting
            if grad.shape != input_tensor.shape:
                ndims_added = grad.ndim - len(input_tensor.shape)
                for _ in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(input_tensor.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
            if input_tensor.grad is None: input_tensor.grad = np.zeros_like(input_tensor.data)
            input_tensor.grad += grad
            
        if other_tensor.requires_grad:
            grad = grad_output
            # Handle broadcasting for other
            if grad.shape != other_tensor.shape:
                ndims_added = grad.ndim - len(other_tensor.shape)
                for _ in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(other_tensor.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
            if other_tensor.grad is None: other_tensor.grad = np.zeros_like(other_tensor.data)
            other_tensor.grad += grad
    
    def sub(self, other):
        """Subtraction operation."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self.add(other.mul(-1))
    
    def mul(self, other):
        """Multiplication operation.""" 
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        result = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        
        if result.requires_grad:
            result._prev = {self, other}
            result.context = (self, other) # store tensors themselves for backward
            result._backward = self._backward_mul

        return result

    def _backward_mul(self, output_tensor: 'Tensor'):
        self_tensor, other_tensor = output_tensor.context
        grad_output = output_tensor.grad

        if self_tensor.requires_grad:
            grad = grad_output * other_tensor.data
            # Handle broadcasting
            if grad.shape != self_tensor.shape:
                ndims_added = grad.ndim - self_tensor.ndim
                for _ in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(self_tensor.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
            if self_tensor.grad is None: self_tensor.grad = np.zeros_like(self_tensor.data)
            self_tensor.grad += grad
        
        if other_tensor.requires_grad:
            grad = grad_output * self_tensor.data
            # Handle broadcasting
            if grad.shape != other_tensor.shape:
                ndims_added = grad.ndim - other_tensor.ndim
                for _ in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(other_tensor.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
            if other_tensor.grad is None: other_tensor.grad = np.zeros_like(other_tensor.data)
            other_tensor.grad += grad

    def div(self, other):
        """Division operation."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self.mul(other.pow(-1))
    
    def matmul(self, other):
        """Matrix multiplication."""
        if not isinstance(other, Tensor):
            raise TypeError("matmul requires Tensor operand")
        
        result = Tensor(np.matmul(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad)

        if result.requires_grad:
            result._prev = {self, other}
            result.context = (self, other)
            result._backward = self._backward_matmul

        return result

    def _backward_matmul(self, output_tensor: 'Tensor'):
        self_tensor, other_tensor = output_tensor.context
        grad_output = output_tensor.grad
        
        if self_tensor.requires_grad:
            if self_tensor.grad is None: self_tensor.grad = np.zeros_like(self_tensor.data)
            self_tensor.grad += np.matmul(grad_output, other_tensor.data.swapaxes(-2, -1))
        
        if other_tensor.requires_grad:
            if other_tensor.grad is None: other_tensor.grad = np.zeros_like(other_tensor.data)
            other_tensor.grad += np.matmul(self_tensor.data.swapaxes(-2, -1), grad_output)
    
    def pow(self, exponent):
        """Power operation."""
        if not isinstance(exponent, (int, float)):
            raise TypeError("pow exponent must be a number")
            
        result = Tensor(np.power(self.data, exponent), requires_grad=self.requires_grad)
        
        if result.requires_grad:
            result._prev = {self}
            result.context = (self, exponent)
            result._backward = self._backward_pow
            
        return result

    def _backward_pow(self, output_tensor: 'Tensor'):
        self_tensor, exponent = output_tensor.context
        grad_output = output_tensor.grad
        
        if self_tensor.requires_grad:
            grad = grad_output * exponent * np.power(self_tensor.data, exponent - 1)
            if self_tensor.grad is None: self_tensor.grad = np.zeros_like(self_tensor.data)
            self_tensor.grad += grad
    
    def sum(self, axis=None, keepdims=False):
        """Sum operation."""
        result = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        
        if result.requires_grad:
            result._prev = {self}
            result.context = self
            result._backward = self._backward_sum
        
        return result

    def _backward_sum(self, output_tensor: 'Tensor'):
        self_tensor = output_tensor.context
        grad_output = output_tensor.grad
        
        if self_tensor.requires_grad:
            if self_tensor.grad is None: self_tensor.grad = np.zeros_like(self_tensor.data)
            self_tensor.grad += np.broadcast_to(grad_output, self_tensor.shape)

    def mean(self, axis=None, keepdims=False):
        """Mean operation."""
        if axis is None:
            size = self.data.size
        else:
            if isinstance(axis, int):
                size = self.data.shape[axis]
            else:
                size = np.prod([self.data.shape[ax] for ax in axis])
        
        return self.sum(axis=axis, keepdims=keepdims) / size
    
    def reshape(self, new_shape):
        """Reshape operation."""
        result = Tensor(self.data.reshape(new_shape), requires_grad=self.requires_grad)
        
        if result.requires_grad:
            result._prev = {self}
            result.context = self
            result._backward = self._backward_reshape
        
        return result

    def _backward_reshape(self, output_tensor: 'Tensor'):
        self_tensor = output_tensor.context
        grad_output = output_tensor.grad
        
        if self_tensor.requires_grad:
            if self_tensor.grad is None: self_tensor.grad = np.zeros_like(self_tensor.data)
            self_tensor.grad += grad_output.reshape(self_tensor.shape)

    def transpose(self, axes=None):
        """Transpose operation."""
        result = Tensor(np.transpose(self.data, axes), requires_grad=self.requires_grad)
        
        if result.requires_grad:
            result._prev = {self}
            result.context = (self, axes)
            result._backward = self._backward_transpose
            
        return result

    def _backward_transpose(self, output_tensor: 'Tensor'):
        self_tensor, axes = output_tensor.context
        grad_output = output_tensor.grad
        
        if self_tensor.requires_grad:
            if axes is None:
                inv_axes = None
            else:
                inv_axes = np.argsort(axes)
            
            if self_tensor.grad is None: self_tensor.grad = np.zeros_like(self_tensor.data)
            self_tensor.grad += np.transpose(grad_output, inv_axes)

    def backward(self, grad: np.ndarray = None):
        if not self.requires_grad:
            return
            
        if grad is None:
            # For the final loss tensor, the gradient is 1
            grad = np.ones_like(self.data, dtype=np.float32)

        if self.grad is None:
            self.grad = np.zeros_like(self.data, dtype=np.float32)
            
        self.grad += grad
        
        # Topologically sort all children in the graph
        topo_order = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo_order.append(v)
        
        build_topo(self)
        
        # Propagate gradients backward through the graph
        for v in reversed(topo_order):
            if v._backward is not _empty_backward:
                v._backward(v)
    
    def zero_grad(self):
        self.grad = None
    
    def detach(self):
        """Detach from computation graph."""
        return Tensor(self.data.copy(), requires_grad=False)
    
    def numpy(self):
        """Convert to numpy array."""
        return self.data.copy()
    
    # Utility methods
    def item(self):
        """Get scalar value."""
        return self.data.item()
    
    @staticmethod
    def zeros(*shape, requires_grad=False):
        """Create tensor of zeros."""
        return Tensor(np.zeros(shape), requires_grad=requires_grad)
    
    @staticmethod 
    def ones(*shape, requires_grad=False):
        """Create tensor of ones."""
        return Tensor(np.ones(shape), requires_grad=requires_grad)
    
    @staticmethod
    def randn(*shape, requires_grad=False):
        """Create tensor with random normal values."""
        return Tensor(np.random.randn(*shape), requires_grad=requires_grad)
    
    @staticmethod
    def rand(*shape, requires_grad=False):
        """Create tensor with random uniform values."""
        return Tensor(np.random.rand(*shape), requires_grad=requires_grad) 