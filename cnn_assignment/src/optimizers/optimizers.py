"""
Optimizer implementations for the CNN framework
"""
import numpy as np
from typing import Dict, List, Optional
from ..core.tensor import Tensor


class Optimizer:
    """
    Base optimizer class.
    """
    
    def __init__(self, learning_rate: float = 0.01):
        """
        Initialize optimizer.
        
        Args:
            learning_rate: Learning rate for parameter updates
        """
        self.learning_rate = learning_rate
        self.iteration = 0
        
    def step(self, parameters: Dict[str, Tensor]):
        """
        Perform one optimization step.
        
        Args:
            parameters: Dictionary of parameters to optimize
        """
        raise NotImplementedError
    
    def zero_grad(self, parameters: Dict[str, Tensor]):
        """
        Zero gradients for all parameters.
        
        Args:
            parameters: Dictionary of parameters
        """
        for param in parameters.values():
            if param.requires_grad:
                param.zero_grad()


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    """
    
    def __init__(self, learning_rate: float = 0.01, weight_decay: float = 0.0):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate: Learning rate
            weight_decay: L2 regularization coefficient
        """
        super().__init__(learning_rate)
        self.weight_decay = weight_decay
        
    def step(self, parameters: Dict[str, Tensor]):
        """
        Perform SGD update.
        
        Args:
            parameters: Dictionary of parameters to update
        """
        for param in parameters.values():
            if param.requires_grad and param.grad is not None:
                # Add weight decay (L2 regularization)
                grad = param.grad
                if self.weight_decay != 0:
                    grad = grad + self.weight_decay * param.data
                
                # Update parameters
                param.data -= self.learning_rate * grad
        
        self.iteration += 1


class Momentum(Optimizer):
    """
    SGD with momentum optimizer.
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9, 
                 weight_decay: float = 0.0):
        """
        Initialize Momentum optimizer.
        
        Args:
            learning_rate: Learning rate
            momentum: Momentum coefficient
            weight_decay: L2 regularization coefficient
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = {}
        
    def step(self, parameters: Dict[str, Tensor]):
        """
        Perform momentum update.
        
        Args:
            parameters: Dictionary of parameters to update
        """
        for name, param in parameters.items():
            if param.requires_grad and param.grad is not None:
                # Initialize velocity if not exists
                if name not in self.velocity:
                    self.velocity[name] = np.zeros_like(param.data)
                
                # Add weight decay
                grad = param.grad
                if self.weight_decay != 0:
                    grad = grad + self.weight_decay * param.data
                
                # Update velocity
                self.velocity[name] = self.momentum * self.velocity[name] + grad
                
                # Update parameters
                param.data -= self.learning_rate * self.velocity[name]
        
        self.iteration += 1


class RMSprop(Optimizer):
    """
    RMSprop optimizer.
    """
    
    def __init__(self, learning_rate: float = 0.001, alpha: float = 0.99, 
                 eps: float = 1e-8, weight_decay: float = 0.0):
        """
        Initialize RMSprop optimizer.
        
        Args:
            learning_rate: Learning rate
            alpha: Smoothing constant
            eps: Small constant for numerical stability
            weight_decay: L2 regularization coefficient
        """
        super().__init__(learning_rate)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.square_avg = {}
        
    def step(self, parameters: Dict[str, Tensor]):
        """
        Perform RMSprop update.
        
        Args:
            parameters: Dictionary of parameters to update
        """
        for name, param in parameters.items():
            if param.requires_grad and param.grad is not None:
                # Initialize square average if not exists
                if name not in self.square_avg:
                    self.square_avg[name] = np.zeros_like(param.data)
                
                # Add weight decay
                grad = param.grad
                if self.weight_decay != 0:
                    grad = grad + self.weight_decay * param.data
                
                # Update square average
                self.square_avg[name] = (self.alpha * self.square_avg[name] + 
                                       (1 - self.alpha) * grad ** 2)
                
                # Update parameters
                param.data -= (self.learning_rate * grad / 
                             (np.sqrt(self.square_avg[name]) + self.eps))
        
        self.iteration += 1


class Adam(Optimizer):
    """
    Adam optimizer.
    """
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0.0):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate: Learning rate
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            eps: Small constant for numerical stability
            weight_decay: L2 regularization coefficient
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}  # First moment estimate
        self.v = {}  # Second moment estimate
        
    def step(self, parameters: Dict[str, Tensor]):
        """
        Perform Adam update.
        
        Args:
            parameters: Dictionary of parameters to update
        """
        self.iteration += 1
        
        for name, param in parameters.items():
            if param.requires_grad and param.grad is not None:
                # Initialize moments if not exists
                if name not in self.m:
                    self.m[name] = np.zeros_like(param.data)
                    self.v[name] = np.zeros_like(param.data)
                
                # Add weight decay
                grad = param.grad
                if self.weight_decay != 0:
                    grad = grad + self.weight_decay * param.data
                
                # Update biased first moment estimate
                self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
                
                # Update biased second raw moment estimate
                self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grad ** 2
                
                # Compute bias-corrected first moment estimate
                m_hat = self.m[name] / (1 - self.beta1 ** self.iteration)
                
                # Compute bias-corrected second raw moment estimate
                v_hat = self.v[name] / (1 - self.beta2 ** self.iteration)
                
                # Update parameters
                param.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)


class AdaGrad(Optimizer):
    """
    AdaGrad optimizer.
    """
    
    def __init__(self, learning_rate: float = 0.01, eps: float = 1e-8, 
                 weight_decay: float = 0.0):
        """
        Initialize AdaGrad optimizer.
        
        Args:
            learning_rate: Learning rate
            eps: Small constant for numerical stability
            weight_decay: L2 regularization coefficient
        """
        super().__init__(learning_rate)
        self.eps = eps
        self.weight_decay = weight_decay
        self.sum_squares = {}
        
    def step(self, parameters: Dict[str, Tensor]):
        """
        Perform AdaGrad update.
        
        Args:
            parameters: Dictionary of parameters to update
        """
        for name, param in parameters.items():
            if param.requires_grad and param.grad is not None:
                # Initialize sum of squares if not exists
                if name not in self.sum_squares:
                    self.sum_squares[name] = np.zeros_like(param.data)
                
                # Add weight decay
                grad = param.grad
                if self.weight_decay != 0:
                    grad = grad + self.weight_decay * param.data
                
                # Accumulate squared gradients
                self.sum_squares[name] += grad ** 2
                
                # Update parameters
                param.data -= (self.learning_rate * grad / 
                             (np.sqrt(self.sum_squares[name]) + self.eps))
        
        self.iteration += 1


class LearningRateScheduler:
    """
    Learning rate scheduler for optimizers.
    """
    
    def __init__(self, optimizer: Optimizer, schedule_type: str = 'step', **kwargs):
        """
        Initialize learning rate scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            schedule_type: Type of schedule ('step', 'exponential', 'cosine')
            **kwargs: Schedule-specific parameters
        """
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        self.initial_lr = optimizer.learning_rate
        self.kwargs = kwargs
        
    def step(self, epoch: int):
        """
        Update learning rate based on epoch.
        
        Args:
            epoch: Current epoch number
        """
        if self.schedule_type == 'step':
            step_size = self.kwargs.get('step_size', 30)
            gamma = self.kwargs.get('gamma', 0.1)
            self.optimizer.learning_rate = self.initial_lr * (gamma ** (epoch // step_size))
            
        elif self.schedule_type == 'exponential':
            gamma = self.kwargs.get('gamma', 0.95)
            self.optimizer.learning_rate = self.initial_lr * (gamma ** epoch)
            
        elif self.schedule_type == 'cosine':
            T_max = self.kwargs.get('T_max', 100)
            eta_min = self.kwargs.get('eta_min', 0)
            self.optimizer.learning_rate = (eta_min + 
                (self.initial_lr - eta_min) * (1 + np.cos(np.pi * epoch / T_max)) / 2)


class EarlyStopping:
    """
    Early stopping mechanism for training.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore best weights on stop
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model_parameters: Dict[str, Tensor]) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model_parameters: Model parameters to potentially save
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                # Save best weights
                self.best_weights = {name: param.data.copy() 
                                   for name, param in model_parameters.items()}
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                # Restore best weights
                for name, param in model_parameters.items():
                    if name in self.best_weights:
                        param.data = self.best_weights[name].copy()
            return True
            
        return False 