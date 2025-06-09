#!/usr/bin/env python3
"""
LFW Face Recognition Trainer
============================

This script trains a Convolutional Neural Network (CNN) on the
Labeled Faces in the Wild (LFW) dataset, saves the trained model,
and evaluates its performance by generating a confusion matrix
and a classification report.
"""
import numpy as np
import sys
import os
import pickle
from tqdm import tqdm

# Add project root to path for absolute imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, os.path.dirname(project_root))

from src.core.layer import Sequential
from src.core.tensor import Tensor
from src.layers.conv import Conv2D
from src.layers.pooling import MaxPooling2D
from src.layers.dense import Dense
from src.layers.flatten import Flatten
from src.optimizers.optimizers import Adam
from src.utils.data_utils import load_lfw_dataset
from src.utils.eval_utils import evaluate_model

# --- Model Definition ---
def LFWNet(n_classes, input_shape):
    """Defines the CNN architecture for LFW dataset."""
    model = Sequential(
        Conv2D(out_channels=32, kernel_size=3, padding=1, activation='relu', input_shape=input_shape, name='conv1'),
        MaxPooling2D(pool_size=2, name='pool1'),
        Conv2D(out_channels=64, kernel_size=3, padding=1, activation='relu', name='conv2'),
        MaxPooling2D(pool_size=2, name='pool2'),
        Conv2D(out_channels=128, kernel_size=3, padding=1, activation='relu', name='conv3'),
        MaxPooling2D(pool_size=2, name='pool3'),
        Flatten(name='flatten'),
        Dense(units=512, activation='relu', name='fc1'),
        Dense(units=n_classes, activation='softmax', name='output')
    )
    return model

# --- Loss Function ---
def cross_entropy_loss(predictions, targets):
    """Computes cross-entropy loss and its gradient."""
    epsilon = 1e-9
    predictions_clipped = np.clip(predictions.data, epsilon, 1 - epsilon)
    
    # Loss calculation
    loss_data = -np.sum(targets * np.log(predictions_clipped)) / predictions.shape[0]
    loss = Tensor(np.array(loss_data), requires_grad=True)
    
    # Gradient for backward pass
    grad = (predictions_clipped - targets) / predictions.shape[0]
    
    # Define backward function for the loss tensor
    def _backward():
        predictions.backward(grad)
    
    loss._backward = _backward
    loss._prev = {predictions}
    
    return loss

# --- Utility Functions ---
def one_hot_encode(labels, num_classes):
    """Converts integer labels to one-hot encoding."""
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

def save_model(model, filepath):
    """Saves the model state to a file using pickle."""
    print(f"\nSaving model to {filepath}...")
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully.")

def load_model(filepath):
    """Loads a model state from a file."""
    if not os.path.exists(filepath):
        print(f"No model file found at {filepath}.")
        return None
    print(f"Loading model from {filepath}...")
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
    return model

# --- Main Training and Evaluation ---
def main():
    """Main function to run the training and evaluation pipeline."""
    # --- 1. Load Data ---
    data = load_lfw_dataset(min_faces_per_person=60, resize=0.5)
    if data is None:
        return
    X_train, y_train, X_test, y_test, target_names, n_classes = data
    
    # --- 2. Initialize Model, Optimizer, and Hyperparameters ---
    input_shape = X_train.shape[1:]
    model = LFWNet(n_classes, input_shape)
    optimizer = Adam(learning_rate=0.001)
    
    epochs = 15
    batch_size = 32
    
    print("\n" + "="*20 + " Starting Training " + "="*20)
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, Optimizer: Adam")
    model.summary()
    
    # --- 3. Training Loop ---
    num_samples = X_train.shape[0]
    for epoch in range(epochs):
        epoch_loss = 0
        
        # Shuffle training data
        permutation = np.random.permutation(num_samples)
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]
        
        progress_bar = tqdm(range(0, num_samples, batch_size), desc=f"Epoch {epoch+1}/{epochs}")
        for i in progress_bar:
            # Get batch
            X_batch = Tensor(X_train_shuffled[i:i+batch_size])
            y_batch_int = y_train_shuffled[i:i+batch_size]
            y_batch_one_hot = one_hot_encode(y_batch_int, n_classes)
            
            # Forward pass
            predictions = model.forward(X_batch)
            
            # Compute loss
            loss = cross_entropy_loss(predictions, y_batch_one_hot)
            epoch_loss += loss.data
            
            # Backward pass
            model.backward(loss)
            
            # Update weights
            optimizer.step(model.parameters())
            
            progress_bar.set_postfix({'loss': loss.data.item()})
            
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {epoch_loss / (num_samples / batch_size):.4f}")

    print("="*21 + " Training Finished " + "="*21 + "\n")

    # --- 4. Save the Trained Model ---
    save_model(model, "lfw_model.pkl")

    # --- 5. Evaluate the Model ---
    print("Evaluating model on the test set...")
    # Get predictions in batches to avoid memory issues
    test_predictions = []
    for i in range(0, X_test.shape[0], batch_size):
        X_batch = Tensor(X_test[i:i+batch_size])
        preds = model.forward(X_batch)
        test_predictions.append(preds.data)
    
    test_predictions = np.vstack(test_predictions)
    
    evaluate_model(y_test, test_predictions, target_names, results_dir="lfw_results")
    
if __name__ == "__main__":
    main() 