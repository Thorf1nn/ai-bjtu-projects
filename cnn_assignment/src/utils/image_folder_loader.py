import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_from_image_folder(root_dir, image_size=(64, 64), test_size=0.2, random_state=42):
    """
    Loads a dataset from a root directory structured with class-named subfolders.

    Assumes a directory structure like:
    - root_dir/
        - class_1/
            - image_1.jpg
            - image_2.png
        - class_2/
            - image_3.jpeg
    
    Args:
        root_dir (str): The path to the root directory of the dataset.
        image_size (tuple): The target size (height, width) to resize images to.
        test_size (float): The proportion of the dataset to use for the test split.
        random_state (int): The seed for the random number generator for reproducibility.

    Returns:
        tuple: A tuple containing (X_train, y_train, X_test, y_test, class_names).
               Returns None if the directory is invalid.
    """
    if not os.path.isdir(root_dir):
        logger.error(f"Error: Directory not found at '{root_dir}'")
        return None

    images = []
    labels = []
    class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    logger.info(f"Found {len(class_names)} classes: {class_names}")

    for class_name in tqdm(class_names, desc="Loading images"):
        class_dir = os.path.join(root_dir, class_name)
        class_idx = class_to_idx[class_name]
        
        # Guard against non-directory items, just in case
        if not os.path.isdir(class_dir):
            continue

        for img_name in os.listdir(class_dir):
            try:
                img_path = os.path.join(class_dir, img_name)
                
                # Skip if the path is a directory
                if os.path.isdir(img_path):
                    continue

                # Open image, convert to RGB, and resize
                img = Image.open(img_path).convert('RGB').resize(image_size[::-1]) # PIL uses (w, h)
                img_array = np.array(img, dtype=np.float32)
                
                # Transpose to (channels, height, width) and normalize
                img_array = img_array.transpose(2, 0, 1) / 255.0
                
                images.append(img_array)
                labels.append(class_idx)
            except Exception as e:
                logger.warning(f"Could not load image {img_name} in {class_name}: {e}")

    if not images:
        logger.error("No images were loaded. Please check the directory structure and image files.")
        return None
        
    X = np.array(images)
    y = np.array(labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info("Dataset loaded successfully from image folder.")
    logger.info(f"  - Total images loaded: {len(X)}")
    logger.info(f"  - Training samples: {len(X_train)}")
    logger.info(f"  - Test samples: {len(X_test)}")
    logger.info(f"  - Image shape (C, H, W): {X_train.shape[1:]}")

    return X_train, y_train, X_test, y_test, class_names 