import numpy as np
import pickle
from src.layers import Dense
from src.losses import get_loss, get_regularizer
from src.optimizers import get_optimizer

class MLP:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None
        self.regularizer = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, optimizer, loss, l1=0, l2=0):
        self.optimizer = get_optimizer(optimizer)
        self.loss = get_loss(loss)
        self.regularizer = get_regularizer(l1, l2)

    def _forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def _backward(self, y_true, y_pred):
        grad = self.loss.derivative(y_true, y_pred)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def _update_weights(self):
        for layer in self.layers:
            if isinstance(layer, Dense):
                if self.regularizer:
                    layer.grad_weights += self.regularizer.derivative(layer.weights)
                self.optimizer.update(layer)
    
    def fit(self, X_train, y_train, epochs, batch_size, validation_data=None, early_stopping_patience=None):
        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Shuffle training data
            permutation = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]
            
            epoch_loss = 0
            for i in range(0, X_train.shape[0], batch_size):
                x_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]

                # Forward pass
                y_pred = self._forward(x_batch)

                # Compute loss
                loss = self.loss(y_batch, y_pred)
                if self.regularizer:
                    reg_loss = sum(self.regularizer(l.weights) for l in self.layers if isinstance(l, Dense))
                    loss += reg_loss
                epoch_loss += loss

                # Backward pass
                self._backward(y_batch, y_pred)

                # Update weights
                self._update_weights()
            
            epoch_loss /= (X_train.shape[0] / batch_size)
            history['loss'].append(epoch_loss)

            # Validation
            if validation_data:
                X_val, y_val = validation_data
                y_val_pred = self.predict(X_val)
                val_loss = self.loss(y_val, y_val_pred)
                history['val_loss'].append(val_loss)

                # Accuracy for classification
                if self.loss.__class__.__name__ == 'CategoricalCrossentropy':
                    train_acc = self._accuracy(self.predict(X_train), y_train)
                    val_acc = self._accuracy(y_val_pred, y_val)
                    history['accuracy'].append(train_acc)
                    history['val_accuracy'].append(val_acc)
                    print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - accuracy: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")

                    # Early stopping
                    if early_stopping_patience is not None:
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            print("Early stopping triggered.")
                            break
                else: # Regression
                    print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}")

        return history

    def predict(self, x):
        return self._forward(x)
        
    def _accuracy(self, y_pred, y_true):
        return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
