import torch
import torch.nn as nn

layer = nn.Linear(2, 1)

x = torch.tensor([[10.0, 5.0]])
y_actual = torch.tensor([[15.0]])

# The optimizer needs to know which numbers it's allowed to adjust (layer.parameters())
# and how big of a step to take each time (lr = learning rate)
optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)

print("Weights BEFORE update:", layer.weight)
print("Bias BEFORE update:", layer.bias)

y_pred = layer(x)
criterion = nn.MSELoss()
loss = criterion(y_pred, y_actual)
print("\nLoss:", loss.item())

loss.backward()

# This is the new line. It applies the formula:
# w_new = w_old - lr * gradient
optimizer.step()

print("\nWeights AFTER update:", layer.weight)
print("Bias AFTER update:", layer.bias)