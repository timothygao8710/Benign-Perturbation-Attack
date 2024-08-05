import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Generate data
def generate_data(num_samples):
    data = []
    labels = []
    for _ in range(num_samples):
        sequence = np.random.randint(1, 101, size=3)
        if sequence[2] > sequence[1] and sequence[1] > sequence[0]:
            label = 1
        else:
            label = 0

        data.append(sequence)
        labels.append(label)
    return np.array(data), np.array(labels)

# Generate training and testing data
X_train, y_train = generate_data(800)
X_test, y_test = generate_data(4000)

# Normalize the data and convert to PyTorch tensors
X_train = torch.FloatTensor(X_train) / 100.0
y_train = torch.FloatTensor(y_train).view(-1, 1)  # Reshape to (n, 1)
X_test = torch.FloatTensor(X_test) / 100.0
y_test = torch.FloatTensor(y_test).view(-1, 1)  # Reshape to (n, 1)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

# Initialize the model, loss function, and optimizer
model = MLP()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    test_accuracy = ((test_outputs > 0.5) == y_test).float().mean()
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Make predictions
predictions = (model(X_test) > 0.5).int()

# Print some example predictions
for i in range(10):
    print(f"Sequence: {X_test[i] * 100}, True Label: {y_test[i].item()}, Predicted: {predictions[i].item()}")