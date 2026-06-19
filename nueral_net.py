import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data (same as before)
df = pd.read_csv("weather.csv")
df["date"] = pd.to_datetime(df["date"])
df["temp_tomorrow"] = df["temp_max"].shift(-1)
df = df.dropna()

X = df[["temp_max", "temp_min", "precipitation", "windspeed"]].values
y = df["temp_tomorrow"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Neural nets train better when inputs are scaled to similar ranges
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert everything to PyTorch tensors
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define the model: this single Linear layer is mathematically identical
# to the linear regression you just ran
model = nn.Linear(4, 1)

# Loss function: mean squared error, same thing R² was built on
criterion = nn.MSELoss()

# Optimizer: this is the algorithm that adjusts the weights to reduce loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 200
losses = []

for epoch in range(epochs):
    # Forward pass: make predictions
    y_pred = model(X_train_t)
    loss = criterion(y_pred, y_train_t)

    # Backward pass: compute gradients and update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Evaluate on test set
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_t)
    test_loss = criterion(y_pred_test, y_test_t)
    mae = torch.mean(torch.abs(y_pred_test - y_test_t))

print(f"\nTest MSE: {test_loss.item():.4f}")
print(f"Test MAE: {mae.item():.2f}°C")

# Plot loss curve
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.title("Training Loss Over Time")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.savefig("training_loss.png")
plt.show()