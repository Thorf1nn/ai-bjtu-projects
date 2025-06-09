"""
Simple CNN Example using our custom framework
"""
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the framework modules
import numpy as np
from core.tensor import Tensor
from layers.conv import Conv2D


def main():
    """
    Demonstrate basic CNN operations.
    """
    print("CNN Framework Demo")
    print("=" * 50)
    
    # Create sample input data (batch_size=2, channels=3, height=32, width=32)
    np.random.seed(42)
    input_data = np.random.randn(2, 3, 32, 32).astype(np.float32)
    x = Tensor(input_data, requires_grad=True)
    
    print(f"Input shape: {x.shape}")
    
    # Create a simple CNN layer
    conv1 = Conv2D(out_channels=16, kernel_size=3, stride=1, padding=1, 
                   activation='relu', name='conv1')
    
    # Build the layer
    conv1.build(x.shape)
    print(f"Conv1 output shape: {conv1.output_shape}")
    print(f"Conv1 parameters: {conv1.count_parameters()}")
    
    # Forward pass
    print("\nPerforming forward pass...")
    y = conv1.forward(x)
    print(f"Output shape: {y.shape}")
    print(f"Output mean: {np.mean(y.data):.4f}")
    print(f"Output std: {np.std(y.data):.4f}")
    
    # Simple loss (mean squared error)
    target = Tensor(np.ones_like(y.data) * 0.5, requires_grad=False)
    loss = ((y - target) ** 2).mean()
    print(f"\nLoss: {loss.item():.4f}")
    
    # Backward pass
    print("\nPerforming backward pass...")
    loss.backward()
    
    # Check gradients
    print(f"Input gradient shape: {x.grad.shape}")
    print(f"Input gradient mean: {np.mean(x.grad):.6f}")
    
    print(f"Weight gradient shape: {conv1.weight.grad.shape}")
    print(f"Weight gradient mean: {np.mean(conv1.weight.grad):.6f}")
    
    if conv1.use_bias:
        print(f"Bias gradient shape: {conv1.bias.grad.shape}")
        print(f"Bias gradient mean: {np.mean(conv1.bias.grad):.6f}")
    
    print("\nDemo completed successfully!")
    
    # Test activation functions
    print("\n" + "=" * 50)
    print("Testing Activation Functions")
    print("=" * 50)
    
    test_input = Tensor(np.array([-2, -1, 0, 1, 2]).astype(np.float32), requires_grad=True)
    
    # Test ReLU
    relu_output = relu(test_input)
    print(f"Input: {test_input.data}")
    print(f"ReLU output: {relu_output.data}")
    
    # Test backward pass
    grad_output = Tensor(np.ones_like(relu_output.data))
    relu_output.backward(grad_output.data)
    print(f"ReLU gradient: {test_input.grad}")


if __name__ == "__main__":
    main() 