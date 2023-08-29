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

# %% Setting Archticture network

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc0 = nn.Linear(input_size, 512)
        self.bn0 = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, output_size)
        # Initialize the weights using Kaiming (He) Normal initialization for ReLU
        self.init_weights()

    def forward(self, x):
        x = F.leaky_relu(self.bn0(self.fc0(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.01)
        x = torch.tanh(self.fc4(x))
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
batch_size = 256
file_path_x = os.path.join("Numpy_array_save", "train_OneRank", "x_train.npy")
file_path_y = os.path.join("Numpy_array_save", "train_OneRank", "y_train.npy")
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
input_size = 5  # Size of each input sample
N = 5 # Number of players
lr_init = 0.01
output_size = N ** 2  # Size of output (vectorized Q matrix)
model = QNetwork(input_size, output_size)
criterion = nn.MSELoss()  # Mean squared error loss
optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=0.9)
scheduler = StepLR(optimizer, step_size=500, gamma=0.5)

# %% Training loop:
num_epochs = 1000
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
PATH = './Q_Net.pth'
torch.save(model.state_dict(), PATH)
print("Finsh")