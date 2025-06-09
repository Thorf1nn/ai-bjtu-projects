import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(y_true, y_pred, class_names, filepath=None):
    """
    Plots, displays, and optionally saves the confusion matrix.

    Args:
        y_true (np.ndarray): Array of true labels.
        y_pred (np.ndarray): Array of predicted labels.
        class_names (list): List of class names for axis ticks.
        filepath (str, optional): Path to save the plot image. If None,
            the plot is displayed. Defaults to None.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                          xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    
    if filepath:
        print(f"Saving confusion matrix to {filepath}...")
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        print("Confusion matrix saved.")
    else:
        plt.show()

def evaluate_model(y_true, y_pred_probs, class_names, results_dir="results"):
    """
    Generates and saves a classification report and a confusion matrix plot.

    Args:
        y_true (np.ndarray): Array of true labels.
        y_pred_probs (np.ndarray): Array of predicted probabilities from the model.
        class_names (list): List of class names for the report and plot.
        results_dir (str): Directory to save the evaluation artifacts.
                           Defaults to "results".
    """
    # Convert probabilities to class predictions
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n" + "="*25 + " Model Evaluation " + "="*25)
    
    # Generate and print the classification report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Save the report to a file
    report_path = os.path.join(results_dir, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("="*len("Classification Report") + "\n")
        f.write(report)
    print(f"Classification report saved to {report_path}")

    # Plot and save the confusion matrix
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, class_names, filepath=cm_path)
    print("="*68 + "\n") 