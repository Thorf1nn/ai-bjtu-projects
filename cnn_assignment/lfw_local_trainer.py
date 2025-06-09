#!/usr/bin/env python3
"""
LFW Local Face Recognition Trainer
==================================

This script trains a CNN on a local copy of the LFW dataset,
saves the trained model, and evaluates its performance. It uses
the generic `load_from_image_folder` utility.
"""
import numpy as np
import sys
import os
import pickle
from tqdm import tqdm
import shutil
import csv
import matplotlib.pyplot as plt

# Add project root to path for absolute imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    # In this script, the project root is one level up
    sys.path.insert(0, os.path.dirname(project_root))

from src.core.layer import Sequential
from src.core.tensor import Tensor
from src.layers.conv import Conv2D
from src.layers.pooling import MaxPooling2D
from src.layers.dense import Dense
from src.layers.flatten import Flatten
from src.optimizers.optimizers import Adam
from src.utils.image_folder_loader import load_from_image_folder
from src.utils.eval_utils import evaluate_model

# --- Model Definition ---
def LFWNet(n_classes):
    """Defines the CNN architecture for LFW dataset."""
    model = Sequential([
        Conv2D(out_channels=32, kernel_size=3, padding=1, activation='relu', name='conv1'),
        MaxPooling2D(pool_size=2, name='pool1'),
        Conv2D(out_channels=64, kernel_size=3, padding=1, activation='relu', name='conv2'),
        MaxPooling2D(pool_size=2, name='pool2'),
        Conv2D(out_channels=128, kernel_size=3, padding=1, activation='relu', name='conv3'),
        MaxPooling2D(pool_size=2, name='pool3'),
        Flatten(name='flatten'),
        Dense(units=512, activation='relu', name='fc1'),
        Dense(units=n_classes, activation='softmax', name='output')
    ])
    return model

# --- Loss Function ---
def cross_entropy_loss(predictions, targets):
    """
    Computes cross-entropy loss and initiates the backward pass.
    Note: The backward pass is now started here.
    """
    epsilon = 1e-9
    # Clip predictions to prevent log(0)
    predictions_clipped = np.clip(predictions.data, epsilon, 1 - epsilon)
    
    # Calculate loss
    loss_data = -np.sum(targets * np.log(predictions_clipped)) / predictions.shape[0]
    loss = Tensor(loss_data) # No longer needs grad itself
    
    # Calculate the initial gradient for the backward pass
    grad_data = (predictions_clipped - targets) / predictions.shape[0]
    
    # Initiate the backward pass from the predictions tensor
    predictions.backward(grad_data)
    
    return loss

# --- Utility Functions ---
def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]

def calculate_accuracy(predictions, targets):
    """Calculates accuracy from predictions and integer targets."""
    predicted_classes = np.argmax(predictions, axis=1)
    return np.mean(predicted_classes == targets)

def save_model(model, filepath):
    print(f"\nSaving model to {filepath}...")
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully.")

def filter_lfw_by_count(source_dir, dest_dir, allnames_csv_path, min_images=20):
    """
    Filters the LFW dataset using the lfw_allnames.csv file to find subjects
    with a minimum number of images, then copies them.
    This avoids listing the entire source directory, which can be slow or fail.
    """
    if os.path.exists(dest_dir):
        # Clean up old filtered data to ensure a fresh run
        print(f"Removing existing filtered directory '{dest_dir}' to ensure clean state.")
        shutil.rmtree(dest_dir)
    
    print(f"Filtering LFW dataset using '{allnames_csv_path}' (min_images_per_person={min_images})...")
    os.makedirs(dest_dir, exist_ok=True)
    
    subjects_to_copy = []
    try:
        with open(allnames_csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) == 2:
                    name, count_str = row
                    try:
                        count = int(count_str)
                        if count >= min_images:
                            subjects_to_copy.append(name)
                    except ValueError:
                        print(f"Warning: Could not parse count for row: {row}")
    except FileNotFoundError:
        print(f"Error: Could not find '{allnames_csv_path}'")
        return

    if not subjects_to_copy:
        print("Warning: No subjects found with the required number of images. The filtered directory will be empty.")
        return

    print(f"Found {len(subjects_to_copy)} subjects to copy.")
    
    for person_name in tqdm(subjects_to_copy, desc="Copying filtered subjects"):
        source_person_dir = os.path.join(source_dir, person_name)
        dest_person_dir = os.path.join(dest_dir, person_name)
        
        if os.path.isdir(source_person_dir):
            shutil.copytree(source_person_dir, dest_person_dir)
        else:
            # This might happen if the CSV is out of sync with the directory
            print(f"Warning: Directory not found for subject '{person_name}' in source directory. Skipping.")
            
    print("Filtering complete.")

def plot_training_history(history, results_dir):
    """Plots and saves the training loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Loss
    ax1.plot(history['loss'], label='Training Loss')
    ax1.set_title('Training Loss vs. Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot Accuracy
    ax2.plot(history['accuracy'], label='Training Accuracy')
    ax2.plot(history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Accuracy vs. Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, "training_history.png")
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    plt.close()

def plot_sample_predictions(model, X, y, class_names, results_dir, num_samples=25):
    """Plots and saves a grid of sample images with their predictions."""
    plt.figure(figsize=(10, 10))
    indices = np.random.choice(len(X), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(5, 5, i + 1)
        img = X[idx]
        # Transpose from (C, H, W) to (H, W, C) for plotting
        img_display = np.transpose(img, (1, 2, 0))
        # Scale to 0-1 if not already
        if img_display.min() < 0 or img_display.max() > 1:
            img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())

        plt.imshow(img_display)
        plt.axis('off')

        true_label = class_names[y[idx]]
        pred_tensor = model.forward(Tensor(img[np.newaxis, ...]))
        pred_label = class_names[np.argmax(pred_tensor.data)]

        title_color = 'green' if true_label == pred_label else 'red'
        plt.title(f"True: {true_label}\\nPred: {pred_label}", color=title_color, fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(results_dir, "sample_predictions.png")
    plt.savefig(save_path)
    print(f"Sample predictions plot saved to {save_path}")
    plt.close()

# --- Main Training and Evaluation ---
def main():
    """Main function to run the training and evaluation pipeline."""
    # --- 1. Filter and Load Data ---
    lfw_source_path = os.path.join(os.path.dirname(__file__), 'data', 'lfw-deepfunneled', 'lfw-deepfunneled')
    lfw_filtered_path = os.path.join(os.path.dirname(__file__), 'data', 'lfw-filtered')
    lfw_allnames_csv = os.path.join(os.path.dirname(__file__), 'data', 'lfw_allnames.csv')
    
    if not os.path.isdir(lfw_source_path) or not os.path.isfile(lfw_allnames_csv):
        print(f"Error: LFW source directory or allnames.csv not found.")
        print(f"  - Searched for dir: '{lfw_source_path}'")
        print(f"  - Searched for file: '{lfw_allnames_csv}'")
        return
        
    filter_lfw_by_count(lfw_source_path, lfw_filtered_path, lfw_allnames_csv, min_images=50)
    
    data = load_from_image_folder(lfw_filtered_path, image_size=(64, 64))
    if data is None:
        return
    X_train, y_train, X_test, y_test, class_names = data
    n_classes = len(class_names)
    
    # --- 2. Initialize Model, Optimizer, and Hyperparameters ---
    model = LFWNet(n_classes)
    optimizer = Adam(learning_rate=0.001)
    
    epochs = 15
    batch_size = 32
    
    print("\n" + "="*20 + " Starting Training on Local LFW " + "="*20)
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, Classes: {n_classes}")
    
    # Build the model with a sample shape to print summary
    sample_input_shape = (batch_size,) + X_train.shape[1:]
    model.summary(input_shape=sample_input_shape)
    
    # --- 3. Training Loop ---
    history = {'loss': [], 'accuracy': [], 'val_accuracy': []}
    num_samples = X_train.shape[0]
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        permutation = np.random.permutation(num_samples)
        X_train_shuffled, y_train_shuffled = X_train[permutation], y_train[permutation]
        
        progress_bar = tqdm(range(0, num_samples, batch_size), desc=f"Epoch {epoch+1}/{epochs}")
        for i in progress_bar:
            X_batch = Tensor(X_train_shuffled[i:i+batch_size])
            y_batch_int = y_train_shuffled[i:i+batch_size]
            y_batch_one_hot = one_hot_encode(y_batch_int, n_classes)
            
            predictions = model.forward(X_batch)
            loss = cross_entropy_loss(predictions, y_batch_one_hot)
            
            # Calculate training accuracy for the batch
            batch_acc = calculate_accuracy(predictions.data, y_batch_int)
            epoch_acc += batch_acc * X_batch.shape[0]
            epoch_loss += loss.data * X_batch.shape[0]
            
            # The backward call is now inside cross_entropy_loss
            # loss.backward() # This is no longer needed
            optimizer.step(model.parameters())
            
            progress_bar.set_postfix({'loss': loss.data.item(), 'acc': batch_acc})
        
        # Calculate and store metrics for the epoch
        avg_epoch_loss = epoch_loss / num_samples
        avg_epoch_acc = epoch_acc / num_samples
        
        # Calculate validation accuracy
        val_predictions = np.vstack([model.forward(Tensor(X_test[i:i+batch_size])).data 
                                     for i in range(0, X_test.shape[0], batch_size)])
        val_acc = calculate_accuracy(val_predictions, y_test)
        
        history['loss'].append(avg_epoch_loss)
        history['accuracy'].append(avg_epoch_acc)
        history['val_accuracy'].append(val_acc)
            
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss:.4f} - Acc: {avg_epoch_acc:.4f} - Val Acc: {val_acc:.4f}")

    print("="*21 + " Training Finished " + "="*21 + "\n")

    # --- 4. Save and Evaluate ---
    model_path = "lfw_local_model.pkl"
    results_dir = "lfw_local_results"
    save_model(model, model_path)
    
    print("Evaluating model on the test set...")
    test_predictions = np.vstack([model.forward(Tensor(X_test[i:i+batch_size])).data 
                                  for i in range(0, X_test.shape[0], batch_size)])
    
    evaluate_model(y_test, test_predictions, class_names, results_dir=results_dir)
    
    # --- 5. Generate and Save Visualizations ---
    plot_training_history(history, results_dir)
    plot_sample_predictions(model, X_test, y_test, class_names, results_dir)
    
if __name__ == "__main__":
    main() 