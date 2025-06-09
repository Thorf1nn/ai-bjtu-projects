#!/usr/bin/env python3
"""
Data Visualization Script
=========================

This script loads a sample of images from a local dataset folder
and displays them in a grid to verify the data is being loaded correctly.
"""
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add project root to path for absolute imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    # Adjust path to be the 'cnn_assignment' directory's parent
    sys.path.insert(0, os.path.dirname(project_root))

from src.utils.image_folder_loader import load_from_image_folder
from cnn_assignment.lfw_local_trainer import filter_lfw_by_count

def visualize_dataset_sample(X, y, class_names, num_images=25):
    """
    Displays a grid of sample images from the dataset.
    
    Args:
        X (np.ndarray): The image data.
        y (np.ndarray): The label data.
        class_names (list): The list of class names.
        num_images (int): The number of images to display.
    """
    plt.figure(figsize=(10, 10))
    
    # Get random indices for sampling
    sample_indices = np.random.choice(len(X), num_images, replace=False)
    
    for i, idx in enumerate(sample_indices):
        plt.subplot(5, 5, i + 1)
        
        # Images are (C, H, W), need to transpose to (H, W, C) for plotting
        img = X[idx].transpose(1, 2, 0)
        
        plt.imshow(img)
        plt.title(class_names[y[idx]])
        plt.axis('off')
        
    plt.suptitle("Sample Images from LFW Dataset", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def main():
    """Main function to load and visualize the data."""
    print("="*20 + " Data Visualization " + "="*20)
    
    # Use the same filtering logic as the trainer to get a good sample
    lfw_source_path = os.path.join(os.path.dirname(__file__), 'data', 'lfw-deepfunneled')
    lfw_filtered_path = os.path.join(os.path.dirname(__file__), 'data', 'lfw-filtered')

    if not os.path.isdir(lfw_source_path):
        print(f"Error: LFW source directory not found at '{lfw_source_path}'")
        print("Please ensure your 'lfw-deepfunneled' folder is inside 'cnn_assignment/data/'")
        return
        
    # Use a less strict filter just for visualization purposes if needed
    filter_lfw_by_count(lfw_source_path, lfw_filtered_path, min_images=20)
    
    print(f"\nLoading data from '{lfw_filtered_path}' for visualization...")
    data = load_from_image_folder(lfw_filtered_path, image_size=(64, 64), test_size=0.9)
    
    if data:
        # We can use either train or test data for visualization
        X_train, y_train, _, _, class_names = data
        print("Data loaded successfully. Preparing visualization...")
        visualize_dataset_sample(X_train, y_train, class_names, num_images=25)
    else:
        print("Failed to load data. Cannot visualize.")

    print("="*54)

if __name__ == "__main__":
    main() 