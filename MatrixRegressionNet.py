# -*- coding: utf-8 -*-
"""
Created on : ------

@author: Ariel_Kantorovich
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import os
from torch.optim.lr_scheduler import StepLR
import torch.nn.init as init

# Define the architecture of the neural network
class MatrixRegressionNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MatrixRegressionNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # BatchNorm layer after the first convolution
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # BatchNorm layer after the second convolution
        self.fc1 = nn.Linear(64 * input_size * input_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        # Initialize the weights using Kaiming (He) Normal initialization for ReLU
        self.init_weights()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension for convolution
        x = self.bn1(torch.relu(self.conv1(x)))  # Apply BatchNorm after the first convolution
        x = self.bn2(torch.relu(self.conv2(x)))  # Apply BatchNorm after the second convolution
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = torch.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # init.normal_(m.weight, mean=0, std=1.0)
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


# %% Read data and prepare to train
batch_size = 128
file_path_x = os.path.join("Numpy_array_save", "train_conv", "x_train.npy")
file_path_y = os.path.join("Numpy_array_save", "train_conv", "y_train.npy")
X_train = np.load(file_path_x)
Y_train = np.load(file_path_y)
# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)

# Split data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)


# Create DataLoader for training and validation sets
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# %% Create the model, loss function, and optimizer:
N = 10 # Number of players
input_size = N  # Size of each input sample
lr_init = 0.01
hidden_size = 64
output_size = N ** 2  # Size of output (vectorized Q matrix)
model = MatrixRegressionNet(input_size, hidden_size, output_size)
criterion = nn.MSELoss()  # Mean squared error loss
optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=0.9)
scheduler = StepLR(optimizer, step_size=200, gamma=1)

# %% Training loop:
num_epochs = 100
train_list = []
valid_list = []
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for inputs, targets in train_loader:
        # Squeeze the input tensor to match the Fc size
        inputs = inputs.squeeze(dim=-1)
        optimizer.zero_grad()
        outputs = model(inputs)
        # Squeeze the target tensor to match the output size
        targets = targets.squeeze(dim=-1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    scheduler.step()  # Update the learning rate

    # Validate
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            # Squeeze the input tensor to match the Fc size
            inputs = inputs.squeeze(dim=-1)
            # Squeeze the target tensor to match the output size
            targets = targets.squeeze(dim=-1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

    train_loss /= len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)
    train_list.append(train_loss)
    valid_list.append(val_loss)
    lr_value = optimizer.param_groups[0]["lr"]
    print(f"Epoch [{epoch+1}/{num_epochs}] Learning Rate:{lr_value} - "
          f"Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}")

# %% Plot and save weights
# Plot results
epochs = np.arange(num_epochs)
plt.plot(epochs, train_list, label='train')
plt.plot(epochs, valid_list, label='valid')
plt.xlabel("# epochs"), plt.ylabel("Loss"), plt.legend()
plt.show()
# Save Network results
PATH = './Matrix_regression_Net.pth'
torch.save(model.state_dict(), PATH)
print("Finsh")