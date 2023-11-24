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
class Wireless_AutoEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Wireless, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.fc6 = nn.Linear(32, output_size)
        self.init_weights()

    def forward(self, x):
        # Encoder Part
        x1 = torch.relu(self.bn1(self.fc1(x)))
        x2 = torch.relu(self.bn2(self.fc2(x1)))
        x3 = torch.relu(self.bn3(self.fc3(x2)))
        # Decoder Part
        x4 = torch.relu(self.bn4(self.fc4(x3))) + x2
        x5 = torch.relu(self.bn5(self.fc5(x4))) + x1

        # Final layer without activation for regression
        output = self.fc6(x5)
        return output

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

class Wireless(nn.Module):
    def __init__(self, input_size, output_size):
        super(Wireless, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, output_size)  # Output layer with 1 neuron for beta prediction
        self.init_weights()

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = torch.relu(self.fc4(x))
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


# %% Parameters
N = 5 # Number of players
input_size = 4
lr_init = 0.03
output_size = 1
Normalize = False
batch_size = 128

# %% Read data and prepare to train
file_path_x = os.path.join("Numpy_array_save", "N=5_wireless", "X_train_new.npy")
file_path_y = os.path.join("Numpy_array_save", "N=5_wireless", "Y_train.npy")
X_train = np.load(file_path_x)
Y_train = np.load(file_path_y)

# Normalize the input data (z-score normalization)
if Normalize:
    epsilon = 1e-10
    mean = X_train.mean(axis=1)
    std = X_train.std(axis=1)
    X_train = (X_train - mean[:, np.newaxis]) / (std[:, np.newaxis] + epsilon)

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Wireless(input_size, output_size).to(device)
criterion = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=lr_init)
scheduler = StepLR(optimizer, step_size=300, gamma=1)

# %% Training loop:
num_epochs = 400
train_list = []
valid_list = []
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for inputs, targets in train_loader:
        # Squeeze the input tensor to match the Fc size
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.squeeze(dim=-1)
        # inputs = inputs.reshape(inputs.shape[0], 1)
        optimizer.zero_grad()
        outputs = model(inputs)
        # Squeeze the target tensor to match the output size
        outputs = outputs.squeeze(dim=-1)
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
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.squeeze(dim=-1)
            # inputs = inputs.reshape(inputs.shape[0], 1)
            outputs = model(inputs)
            outputs = outputs.squeeze(dim=-1)
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
PATH = './Wireless.pth'
torch.save(model.state_dict(), PATH)
print("Finsh")