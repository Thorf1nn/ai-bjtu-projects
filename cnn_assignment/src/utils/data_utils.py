import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_lfw_dataset(min_faces_per_person=70, resize=0.4, test_size=0.25):
    """
    Loads and preprocesses the Labeled Faces in the Wild (LFW) dataset.

    This function downloads the LFW dataset using scikit-learn, filters it
    to include only people with a minimum number of faces, resizes the images,
    and splits it into training and testing sets.

    Args:
        min_faces_per_person (int): The minimum number of pictures per person
            to be included in the dataset.
        resize (float): The ratio to resize the original images.
        test_size (float): The proportion of the dataset to allocate to the test split.

    Returns:
        tuple: A tuple containing:
            - X_train (np.ndarray): Training images.
            - y_train (np.ndarray): Training labels.
            - X_test (np.ndarray): Testing images.
            - y_test (np.ndarray): Testing labels.
            - target_names (list): The names of the target classes.
            - n_classes (int): The total number of classes.
    """
    logger.info(f"Loading LFW dataset (min_faces_per_person={min_faces_per_person}). This may take a moment...")
    
    # Fetch the dataset
    try:
        lfw_people = fetch_lfw_people(
            min_faces_per_person=min_faces_per_person, 
            resize=resize, 
            color=True, 
            slice_=(slice(70, 195), slice(78, 172))
        )
    except Exception as e:
        logger.error(f"Failed to download LFW dataset. Please check your internet connection. Error: {e}")
        return None

    # Get data and target info
    X = lfw_people.images
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    # Reshape data for CNN: (num_samples, channels, height, width)
    # scikit-learn loads color images as (n_samples, h, w, 3), but our CNN
    # expects (n_samples, 3, h, w).
    X = X.transpose(0, 3, 1, 2).astype(np.float32)

    # Normalize pixel values from [0, 255] to [0, 1]
    X /= 255.0

    # Split into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    logger.info("Dataset loaded and preprocessed successfully.")
    logger.info(f"  - Training samples: {X_train.shape[0]}")
    logger.info(f"  - Test samples: {X_test.shape[0]}")
    logger.info(f"  - Image shape (C, H, W): {X_train.shape[1:]}")
    logger.info(f"  - Number of classes: {n_classes}")

    return X_train, y_train, X_test, y_test, target_names, n_classes 