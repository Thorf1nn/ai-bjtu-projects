"""
Basic test of the CNN framework core components
"""
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_tensor():
    """Test basic tensor operations."""
    print("Testing Tensor class...")
    
    # Import here to avoid issues
    from core.tensor import Tensor
    
    # Create tensors
    a = Tensor(np.array([1, 2, 3]), requires_grad=True)
    b = Tensor(np.array([4, 5, 6]), requires_grad=True)
    
    print(f"a: {a}")
    print(f"b: {b}")
    
    # Test addition
    c = a + b
    print(f"a + b: {c}")
    
    # Test multiplication
    d = a * b
    print(f"a * b: {d}")
    
    # Test backward pass
    loss = d.sum()
    print(f"loss: {loss}")
    
    loss.backward()
    print(f"a.grad: {a.grad}")
    print(f"b.grad: {b.grad}")
    
    print("Tensor tests passed!\n")


def test_activations():
    """Test activation functions."""
    print("Testing activation functions...")
    
    from activations.functions import relu, sigmoid
    from core.tensor import Tensor
    
    # Test ReLU
    x = Tensor(np.array([-2, -1, 0, 1, 2]), requires_grad=True)
    y = relu(x)
    
    print(f"Input: {x.data}")
    print(f"ReLU output: {y.data}")
    
    # Test backward
    y.backward(np.ones_like(y.data))
    print(f"ReLU gradient: {x.grad}")
    
    # Reset gradients
    x.zero_grad()
    
    # Test Sigmoid
    y = sigmoid(x)
    print(f"Sigmoid output: {y.data}")
    
    y.backward(np.ones_like(y.data))
    print(f"Sigmoid gradient: {x.grad}")
    
    print("Activation tests passed!\n")


def main():
    """Run all tests."""
    print("CNN Framework Basic Tests")
    print("=" * 50)
    
    try:
        test_tensor()
        test_activations()
        print("All tests passed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 