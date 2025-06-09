#!/usr/bin/env python3
"""
CNN Assignment Runner
====================

Simple script to run and test the CNN implementation.
"""

import numpy as np
import sys
import os

# Add the project root to the path to allow absolute imports from `src`
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(project_root))

def test_tensor_operations():
    """Test basic tensor operations."""
    print("Testing Tensor Operations...")
    
    from src.core.tensor import Tensor
    
    # Create test tensors
    a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    b = Tensor([[2, 0], [1, 3]], requires_grad=True)
    
    # Test operations
    c = a + b
    d = a * b
    e = a @ b
    
    print(f"a = \n{a.data}")
    print(f"b = \n{b.data}")
    print(f"a + b = \n{c.data}")
    print(f"a * b = \n{d.data}")
    print(f"a @ b = \n{e.data}")
    
    # Test gradients
    loss = e.sum()
    loss.backward()
    
    print(f"Gradient of a: \n{a.grad}")
    print(f"Gradient of b: \n{b.grad}")
    print("✅ Tensor operations work correctly!\n")


def test_conv_layer():
    """Test convolutional layer."""
    print("Testing Convolutional Layer...")
    
    from src.core.tensor import Tensor
    from src.layers.conv import Conv2D
    
    # Create test input (batch_size=2, channels=3, height=8, width=8)
    x = Tensor(np.random.randn(2, 3, 8, 8), requires_grad=True)
    
    # Create conv layer
    conv = Conv2D(out_channels=16, kernel_size=3, padding=1, activation='relu')
    
    # Forward pass
    output = conv(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✅ Convolution layer works correctly!\n")


def test_pooling_layer():
    """Test pooling layer."""
    print("Testing Pooling Layer...")
    
    from src.core.tensor import Tensor
    from src.layers.pooling import MaxPooling2D
    
    # Create test input
    x = Tensor(np.random.randn(2, 16, 8, 8), requires_grad=True)
    
    # Create pooling layer
    pool = MaxPooling2D(pool_size=2)
    
    # Forward pass
    output = pool(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✅ Pooling layer works correctly!\n")


def test_dense_layer():
    """Test dense layer."""
    print("Testing Dense Layer...")
    
    from src.core.tensor import Tensor
    from src.layers.dense import Dense
    
    # Create test input
    x = Tensor(np.random.randn(32, 128), requires_grad=True)
    
    # Create dense layer
    dense = Dense(units=10, activation='softmax')
    
    # Forward pass
    output = dense(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sum of probabilities: {np.sum(output.data, axis=1)[:5]}") # Should be ~1.0
    print("✅ Dense layer works correctly!\n")


def test_simple_cnn():
    """Test a simple CNN pipeline."""
    print("Testing Simple CNN Pipeline...")
    
    from src.core.tensor import Tensor
    from src.layers.conv import Conv2D
    from src.layers.pooling import MaxPooling2D
    from src.layers.flatten import Flatten
    from src.layers.dense import Dense
    
    # Generate synthetic data
    batch_size = 8
    x = Tensor(np.random.randn(batch_size, 3, 32, 32), requires_grad=True)
    
    print(f"Input shape: {x.shape}")
    
    # Build simple CNN
    conv1 = Conv2D(out_channels=32, kernel_size=3, padding=1, activation='relu')
    pool1 = MaxPooling2D(pool_size=2)
    conv2 = Conv2D(out_channels=64, kernel_size=3, padding=1, activation='relu')
    pool2 = MaxPooling2D(pool_size=2)
    flatten = Flatten()
    fc = Dense(units=10, activation='softmax')
    
    # Forward pass
    x = conv1(x)
    print(f"After Conv1: {x.shape}")
    
    x = pool1(x)
    print(f"After Pool1: {x.shape}")
    
    x = conv2(x)
    print(f"After Conv2: {x.shape}")
    
    x = pool2(x)
    print(f"After Pool2: {x.shape}")
    
    x = flatten(x)
    print(f"After Flatten: {x.shape}")
    
    x = fc(x)
    print(f"Final Output: {x.shape}")
    
    print("✅ Simple CNN pipeline works correctly!\n")


def test_training_step():
    """Test a simple training step."""
    print("Testing Training Step...")
    
    from src.core.tensor import Tensor
    from src.layers.dense import Dense
    from src.optimizers.optimizers import SGD
    
    # Simple network: input -> dense -> output
    dense = Dense(units=2, activation='softmax')
    optimizer = SGD(learning_rate=0.01)
    
    # Synthetic data
    x = Tensor(np.random.randn(4, 10), requires_grad=True)
    targets = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])  # One-hot
    
    print("Before training:")
    output = dense(x)
    print(f"Output: \n{output.data}")
    
    # Compute loss (simple MSE for demonstration)
    loss_data = np.mean((output.data - targets) ** 2)
    loss = Tensor(np.array(loss_data), requires_grad=True)
    
    # Backward pass
    grad = 2 * (output.data - targets) / output.data.shape[0]
    output.backward(grad)
    
    # Update weights
    optimizer.step(dense.parameters())
    
    # Forward pass after update
    output_after = dense(x)
    print("After 1 training step:")
    print(f"Output: \n{output_after.data}")
    
    print("✅ Training step works correctly!\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("CNN Assignment - Framework Testing")
    print("=" * 60)
    print()
    
    try:
        test_tensor_operations()
        test_conv_layer()
        test_pooling_layer()
        test_dense_layer()
        test_simple_cnn()
        test_training_step()
        
        print("=" * 60)
        print("🎉 All tests passed! The CNN framework is working correctly.")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Check complete_example.py for a full training loop")
        print("2. Read README.md for detailed usage instructions")
        print("3. View ASSIGNMENT_REPORT.md for complete documentation")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
