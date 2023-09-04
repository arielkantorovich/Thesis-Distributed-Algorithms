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


# %% Read data and prepare to test
file_path_x = os.path.join("Numpy_array_save", "test_conv", "x_test.npy")
file_path_y = os.path.join("Numpy_array_save", "test_conv", "y_test.npy")
file_path_Q = os.path.join("Numpy_array_save", "test_conv", "Q.npy")
X_test = np.load(file_path_x)
Y_test = np.load(file_path_y)
Q = np.load(file_path_Q)

# %% Load the trained model
file_path_weights = os.path.join("trains_record", "SGD", "Matrix_regression_Net.pth")
hidden_size = 64
N = 10 # Number of players
input_size = N
output_size = N ** 2  # Size of output (vectorized Q matrix)
model = MatrixRegressionNet(input_size, hidden_size, output_size)
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





