"""
Base Layer class for the CNN framework
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .tensor import Tensor


class Layer(ABC):
    """
    Abstract base class for all layers in the neural network.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize a layer.
        
        Args:
            name: Optional name for the layer
        """
        self.name = name or self.__class__.__name__.lower()
        self.trainable = True
        self.built = False
        self.input_shape = None
        self.output_shape = None
        self._parameters = {}
        self._gradients = {}
        
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
    
    @abstractmethod 
    def backward(self, grad_output: Tensor) -> Tensor:
        """
        Backward pass through the layer.
        
        Args:
            grad_output: Gradient of the loss with respect to the output
            
        Returns:
            Gradient of the loss with respect to the input
        """
        pass
    
    def build(self, input_shape):
        """
        Build the layer (initialize parameters based on input shape).
        
        Args:
            input_shape: Shape of the input tensor
        """
        self.input_shape = input_shape
        self.built = True
    
    def __call__(self, x: Tensor, training: bool = True) -> Tensor:
        """
        Call the layer (forward pass).
        
        Args:
            x: Input tensor
            training: Whether the layer is in training mode
            
        Returns:
            Output tensor
        """
        if not self.built:
            self.build(x.shape)
        
        return self.forward(x)
    
    def parameters(self) -> Dict[str, Tensor]:
        """
        Get all trainable parameters of the layer.
        
        Returns:
            Dictionary of parameter names to tensors
        """
        return {k: v for k, v in self._parameters.items() if v.requires_grad}
    
    def get_parameter(self, name: str) -> Tensor:
        """
        Get a specific parameter by name.
        
        Args:
            name: Parameter name
            
        Returns:
            Parameter tensor
        """
        if name not in self._parameters:
            raise ValueError(f"Parameter '{name}' not found in layer '{self.name}'")
        return self._parameters[name]
    
    def set_parameter(self, name: str, value: Tensor):
        """
        Set a parameter value.
        
        Args:
            name: Parameter name
            value: Parameter tensor
        """
        self._parameters[name] = value
    
    def zero_gradients(self):
        """Zero out all parameter gradients."""
        for param in self._parameters.values():
            if param.requires_grad:
                param.zero_grad()
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the layer.
        
        Returns:
            Configuration dictionary
        """
        return {
            'name': self.name,
            'trainable': self.trainable
        }
    
    def count_parameters(self) -> int:
        """
        Count the total number of trainable parameters.
        
        Returns:
            Number of parameters
        """
        return sum(param.size for param in self.parameters().values())
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class Sequential:
    """
    Sequential container for layers.
    """
    
    def __init__(self, layers: List[Layer] = None):
        """
        Initialize sequential model.
        
        Args:
            layers: List of layers
        """
        self.layers = layers or []
        self.built = False
        
    def add(self, layer: Layer):
        """
        Add a layer to the model.
        
        Args:
            layer: Layer to add
        """
        self.layers.append(layer)
        self.built = False
    
    def forward(self, x: Tensor, training: bool = True) -> Tensor:
        """
        Forward pass through all layers.
        
        Args:
            x: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        if not self.built:
            self.build(x.shape)
        
        for layer in self.layers:
            x = layer(x, training=training)
        return x
    
    def __call__(self, x: Tensor, training: bool = True) -> Tensor:
        """Call the sequential model."""
        return self.forward(x, training=training)
    
    def build(self, input_shape):
        """
        Build all layers.
        
        Args:
            input_shape: Shape of the input
        """
        current_shape = input_shape
        for layer in self.layers:
            if not layer.built:
                layer.build(current_shape)
            # Update current shape for next layer
            # This would need to be implemented per layer type
            current_shape = layer.output_shape if hasattr(layer, 'output_shape') else current_shape
        
        self.built = True
    
    def parameters(self) -> Dict[str, Tensor]:
        """
        Get all parameters from all layers.
        
        Returns:
            Dictionary of all parameters
        """
        params = {}
        for i, layer in enumerate(self.layers):
            layer_params = layer.parameters()
            for name, param in layer_params.items():
                params[f"layer_{i}_{name}"] = param
        return params
    
    def zero_gradients(self):
        """Zero gradients for all layers."""
        for layer in self.layers:
            layer.zero_gradients()
    
    def count_parameters(self) -> int:
        """
        Count total parameters in the model.
        
        Returns:
            Total number of parameters
        """
        return sum(layer.count_parameters() for layer in self.layers)
    
    def __len__(self) -> int:
        return len(self.layers)
    
    def __getitem__(self, idx: int) -> Layer:
        return self.layers[idx]
    
    def summary(self, input_shape=None):
        """
        Prints a summary of the model.
        
        Args:
            input_shape: Optional input shape to build the model if not built.
        """
        if not self.built and input_shape:
            self.build(input_shape)
        
        if not self.built:
            print("Model has not been built yet. Provide an input_shape or run a forward pass.")
            return
            
        print("="*60)
        print(f"Model: Sequential")
        print("="*60)
        print(f"{'Layer (type)':<25} {'Output Shape':<25} {'Param #':<10}")
        print("-"*60)
        
        total_params = 0
        for layer in self.layers:
            layer_type = layer.__class__.__name__
            output_shape = str(getattr(layer, 'output_shape', 'N/A'))
            params = layer.count_parameters()
            total_params += params
            
            print(f"{layer_type:<25} {output_shape:<25} {params:<10,}")

        print("="*60)
        print(f"Total params: {total_params:,}")
        print("="*60)
    
    def __repr__(self) -> str:
        layer_repr = '\n  '.join([repr(layer) for layer in self.layers])
        return f"Sequential([\n  {layer_repr}\n])" 