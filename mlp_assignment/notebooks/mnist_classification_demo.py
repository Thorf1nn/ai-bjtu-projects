import sys
import os
# Add src directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.model import MLP
from src.layers import Dense
from src.utils import load_mnist_from_csv
from src.metrics import accuracy_score, confusion_matrix

def main():
    # 1. Load and prepare data
    print("Loading MNIST data from CSV...")
    (X_train, y_train), (X_test, y_test) = load_mnist_from_csv(data_path='./data')

    # 2. Define the MLP model
    print("Building model...")
    model = MLP()
    model.add(Dense(input_dim=784, output_dim=128, activation='relu', initializer='he_uniform'))
    model.add(Dense(input_dim=128, output_dim=64, activation='relu', initializer='he_uniform'))
    model.add(Dense(input_dim=64, output_dim=10, activation='softmax'))

    # 3. Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', l2=1e-4)

    # 4. Train the model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=64,
        validation_data=(X_test, y_test),
        early_stopping_patience=3
    )

    # 5. Evaluate the model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # 6. Plot history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    # 7. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, num_classes=10)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    # 8. Save the model
    print("Saving model...")
    model.save('mnist_mlp_model.pkl')
    print("Model saved to mnist_mlp_model.pkl")

if __name__ == '__main__':
    main() 