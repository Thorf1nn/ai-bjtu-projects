"""
CNN Framework Implementation from Scratch
=========================================

A complete deep learning framework for Convolutional Neural Networks
implemented without using existing frameworks like PyTorch or TensorFlow.

Author: AI Assistant
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"

from . import core
from . import layers
from . import optimizers
from . import activations
from . import initializers
from . import regularizers
from . import models
from . import utils
from . import losses

__all__ = [
    'core',
    'layers', 
    'optimizers',
    'activations',
    'initializers',
    'regularizers',
    'models',
    'utils',
    'losses'
] 