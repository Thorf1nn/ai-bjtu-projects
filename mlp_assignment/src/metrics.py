import numpy as np

def accuracy_score(y_true, y_pred):
    """
    Calculate accuracy for classification.
    Assumes y_true and y_pred are one-hot encoded.
    """
    return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

def confusion_matrix(y_true, y_pred, num_classes):
    """
    Compute confusion matrix.
    Assumes y_true and y_pred are one-hot encoded.
    """
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true_labels, y_pred_labels):
        cm[true_label, pred_label] += 1
    return cm

def r2_score(y_true, y_pred):
    """
    Calculate R^2 score for regression.
    """
    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - (ss_res / (ss_tot + 1e-8))
