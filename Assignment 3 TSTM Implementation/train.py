def train(model, data, epochs=10, lr=0.01):
    for epoch in range(epochs):
        loss = 0
        for X, Y in data:
            # Forward pass
            outputs = model.forward(X)
            # Compute loss (e.g., MSE or cross-entropy)
            # Backward pass and update (to be implemented)
            pass
        print(f"Epoch {epoch}, Loss: {loss}") 