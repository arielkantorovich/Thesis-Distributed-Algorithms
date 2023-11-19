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
import torch.nn.utils.weight_norm as weight_norm
# %% Setting Archticture network
class Wireless(nn.Module):
    def __init__(self, input_size, output_size):
        super(Wireless, self).__init__()
        self.fc0 = nn.Linear(in_features=input_size, out_features=4096)
        self.bn0 = nn.BatchNorm1d(4096)
        self.fc1 = nn.Linear(in_features=4096, out_features=2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc6 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.fc7 = nn.Linear(64, 32)
        self.bn7 = nn.BatchNorm1d(32)
        self.fc8 = nn.Linear(32, output_size)
        # Initialize the weights using Kaiming (He) Normal initialization for ReLU
        self.init_weights()

    def forward(self, x):
        # Main path
        x = F.relu(self.bn0(self.fc0(x)))
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        x = F.relu(self.bn7(self.fc7(x)))
        x = F.relu(self.fc8(x))
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
file_path_x = os.path.join("Numpy_array_save", "N=10_wireless", "X_test.npy")
file_path_y = os.path.join("Numpy_array_save", "N=10_wireless", "Y_test.npy")

X_test = np.load(file_path_x)
Y_test = np.load(file_path_y)


# %% Load the trained model
file_path_weights = os.path.join("trains_record", "Wireless_example", "Wireless.pth")
N = 10 # Number of players
input_size = 10*N
output_size = N  # Size of output (vectorized Q matrix)
model = Wireless (input_size, output_size)  # Initialize the model with the same architecture
model.load_state_dict(torch.load(file_path_weights, map_location='cpu'))
model.eval()  # Set the model to evaluation mode

# %% Evaluate the model on the test set
# Evaluate the model on the test set
with torch.no_grad():
    X_test = torch.tensor(X_test, dtype=torch.float32)
    # Squeeze the input tensor to match the Fc size
    X_test = X_test.squeeze(dim=-1)
    outputs = model(X_test)
    outputs = outputs.squeeze(dim=-1)
    predicted_values = outputs.numpy()

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





