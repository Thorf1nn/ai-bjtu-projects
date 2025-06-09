import sys
import os
# Add src directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from src.model import MLP
from src.layers import Dense
from src.utils import load_california_housing_data, train_test_split, StandardScaler
from src.metrics import r2_score

def main():
    # 1. Load and prepare data
    print("Loading California Housing data...")
    X, y = load_california_housing_data()

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Define the MLP model
    print("Building model...")
    model = MLP()
    model.add(Dense(input_dim=X_train.shape[1], output_dim=64, activation='relu', initializer='he_uniform'))
    model.add(Dense(input_dim=64, output_dim=32, activation='relu', initializer='he_uniform'))
    model.add(Dense(input_dim=32, output_dim=1, activation='linear'))

    # 3. Compile the model
    model.compile(optimizer='adam', loss='mse')

    # 4. Train the model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=50, # More epochs for regression
        batch_size=32,
        validation_data=(X_test, y_test),
        early_stopping_patience=5
    )

    # 5. Evaluate the model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"Test R^2 Score: {r2:.4f}")

    # 6. Plot history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # 7. Plot predictions vs actual
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    plt.title('Predictions vs. Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    plt.tight_layout()
    plt.show()
    
    # 8. Save the model
    print("Saving model...")
    model.save('regression_mlp_model.pkl')
    print("Model saved to regression_mlp_model.pkl")

if __name__ == '__main__':
    main() 