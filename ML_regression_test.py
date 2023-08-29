# -*- coding: utf-8 -*-
"""
Created on : ------

@author: Ariel_Kantorovich
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.nn.init as init
# %% Setting Archticture network
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc0 = nn.Linear(input_size, 512)
        self.bn0 = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = self.bn1 = nn.BatchNorm1d(256)
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

# %% Read data and prepare to test
file_path_x = os.path.join("Numpy_array_save", "test_OneRank", "x_test.npy")
file_path_y = os.path.join("Numpy_array_save", "test_OneRank", "y_test.npy")
file_path_Q = os.path.join("Numpy_array_save", "test_OneRank", "Q.npy")
X_test = np.load(file_path_x)
Y_test = np.load(file_path_y)
Q = np.load(file_path_Q)

# %% Load the trained model
file_path_weights = os.path.join("trains_record", "SGD_withMomentum_1000Epochs(One_rank)", "Q_Net.pth")
input_size = 5
N = 5 # Number of players
output_size = N ** 2  # Size of output (vectorized Q matrix)
model = QNetwork(input_size, output_size)  # Initialize the model with the same architecture
model.load_state_dict(torch.load(file_path_weights))
model.eval()  # Set the model to evaluation mode

# %% Evaluate the model on the test set
# Evaluate the model on the test set
with torch.no_grad():
    X_test = torch.tensor(X_test, dtype=torch.float32)
    # Squeeze the input tensor to match the Fc size
    X_test = X_test.squeeze(dim=-1)
    outputs = model(X_test)
    predicted_values = outputs.numpy()

# Reshape predicted_values and y_test to match your desired shape
predicted_values = predicted_values.reshape(-1, N**2, 1)
Y_test = Y_test.reshape(-1, N**2, 1)

# Calculate some evaluation metrics (you can customize this part)
mse = ((predicted_values - Y_test)**2).mean()
print("Mean Squared Error:", mse)

# Visualize the results with different colors for actual and predicted values
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, predicted_values, c='blue', label='Predict Values', alpha=0.5)
plt.scatter(Y_test, Y_test, c='red', label='Actual Values', alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.legend()
plt.show()

print("Finsh !!!!")





