"""
Created on : ------

@author: Ariel_Kantorovich
"""
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import torch.nn.init as init
import os
import torch.nn.functional as F
from sklearn.preprocessing import PolynomialFeatures


class SimpleNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNet, self).__init__()
        # Define layers
        self.linear1 = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(128, output_size)
        self.init_weights()

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
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


class DynamicQuadraticNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(DynamicQuadraticNet, self).__init__()
        # Define layers
        self.linear1 = nn.Linear(input_size, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.attention1 = nn.Linear(4096, 1, bias=False)

        self.linear2 = nn.Linear(4096, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.attention2 = nn.Linear(1024, 1, bias=False)

        self.linear3 = nn.Linear(1024, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.attention3 = nn.Linear(32, 1, bias=False)

        self.linear5 = nn.Linear(32, output_size)

    def forward(self, x):
        # Linear transformation
        x = torch.relu(self.bn1(self.linear1(x)))
        # Attention mechanism
        attention_scores_1 = torch.softmax(self.attention1(x), dim=1)
        x = attention_scores_1 * x

        x = torch.relu(self.bn2(self.linear2(x)))
        attention_scores_2 = torch.softmax(self.attention2(x), dim=1)
        x = attention_scores_2 * x

        x = torch.relu(self.bn3(self.linear3(x)))
        attention_scores_3 = torch.softmax(self.attention3(x), dim=1)
        x = attention_scores_3 * x

        # Output
        x = torch.relu(self.linear5(x))

        return x


class Congestion(nn.Module):
    def __init__(self, input_size, output_size):
        super(Congestion, self).__init__()
        self.fc00 = nn.Linear(input_size, 256)
        self.bn00 = nn.BatchNorm1d(256)
        self.fc0 = nn.Linear(256, 128)
        self.bn0 = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(128, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, output_size)

        self.init_weights()

    def forward(self, x):
        x = F.relu(self.bn00(self.fc00(x)))
        x = F.relu(self.bn0(self.fc0(x)))
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.fc4(x))
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

class Shallow_Congestion(nn.Module):
    def __init__(self, input_size, output_size):
        super(Shallow_Congestion, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, output_size)  # Output layer with 1 neuron for beta prediction
        self.init_weights()

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.fc4(x))
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

def Generate_Game_Param(L, K):
    """
    :param K:(int), Number of strategy in game
    :return:
    """
    A = np.random.rand(L, K, )
    B = np.random.rand(L, K, )
    C = np.random.rand(L, K, )
    return A, B, C


def Initialize_action(L, N, P):
    """
    This function build action Matrix each player how much budget is spend on path Pi
    :param N: (int) number of players
    :param P: (int) number of paths from source to targer
    :return: Xe (2D-array) size NxP
    """
    Xn_k = np.random.rand(L, N, P)
    row_sums = np.sum(Xn_k, axis=2, keepdims=True)
    Xn_k /= row_sums
    return Xn_k

# Hyper Parameters
prob = 0.1
nodes = 20
N = 20 # Number of players
T = 500
L = 200
K = 4 # Number of strategy


# # Generate A, B, C Parameters
# A, B, C = Generate_Game_Param(L, K)
#
# # Generate action of players
# nash_Xn_k = Initialize_action(L, N, K)
# # Prepare data to Network
# Xe_nash = np.sum(nash_Xn_k, axis=1)
# C_k_nash = A + B * Xe_nash + C * (Xe_nash ** 2)
#
# # Add more samples
# Xn_k_0 = Initialize_action(L, N, K)
# Xe_0 = np.sum(Xn_k_0, axis=1)
# C_k_0 = A + B * Xe_0 + C * (Xe_0 ** 2)
#
# Xn_k_1 = Initialize_action(L, N, K)
# Xe_1 = np.sum(Xn_k_1, axis=1)
# C_k_1 = A + B * Xe_1 + C * (Xe_1 ** 2)
####################################################################
# Xn_k_2 = Initialize_action(L, N, K)
# Xe_2 = np.sum(Xn_k_2, axis=1)
# C_k_2 = A + B * Xe_2 + C * (Xe_2 ** 2)
# Xn_k_3 = Initialize_action(L, N, K)
# Xe_3 = np.sum(Xn_k_3, axis=1)
# C_k_3 = A + B * Xe_3 + C * (Xe_3 ** 2)
# Xn_k_4 = Initialize_action(L, N, K)
# Xe_4 = np.sum(Xn_k_4, axis=1)
# C_k_4 = A + B * Xe_4 + C * (Xe_4 ** 2)
#
# Xn_k_5 = Initialize_action(L, N, K)
# Xe_5 = np.sum(Xn_k_5, axis=1)
# C_k_5 = A + B * Xe_5 + C * (Xe_5 ** 2)
#
# Xn_k_6 = Initialize_action(L, N, K)
# Xe_6 = np.sum(Xn_k_6, axis=1)
# C_k_6 = A + B * Xe_6 + C * (Xe_6 ** 2)
#
# Xn_k_7 = Initialize_action(L, N, K)
# Xe_7 = np.sum(Xn_k_7, axis=1)
# C_k_7 = A + B * Xe_7 + C * (Xe_7 ** 2)
#
# Xn_k_8 = Initialize_action(L, N, K)
# Xe_8 = np.sum(Xn_k_8, axis=1)
# C_k_8 = A + B * Xe_8 + C * (Xe_8 ** 2)

# Build Xtest
# X_Poly = np.concatenate([Xe_nash, Xe_0, Xe_1], axis=-1)
# poly = PolynomialFeatures(2)
# X = poly.fit_transform(X_Poly)
# X_test = np.concatenate([C_k_nash, C_k_0, C_k_1, X], axis=-1)

Xn_k = Initialize_action(L, N, K)
X_test = np.sum(Xn_k, axis=1)

A = np.random.rand(1)
B = np.random.rand(1)
C = np.random.rand(1)

Ck = A + B * X_test + C * (X_test ** 2)


# X_test = np.concatenate([C_k_nash, Xe_nash,
#                          C_k_0, Xe_0,
#                          C_k_1, Xe_1,
#                          C_k_2, Xe_2,
#                          C_k_2, Xe_2,
#                          C_k_4, Xe_4,
#                          C_k_5, Xe_5,
#                          C_k_6, Xe_6,
#                          C_k_7, Xe_7,
#                          C_k_8, Xe_8], axis=-1)

# file_path_Xn_k = os.path.join("Numpy_array_save", "X_train(Check_F32).npy")
# file_path_Ck = os.path.join("Numpy_array_save", "Y_train(Check_F32).npy")
# X_test = np.load(file_path_Xn_k)
# Ck = np.load(file_path_Ck)

# Get Beta from Neural Network
device = ["cuda" if torch.cuda.is_available() else "cpu"]
file_path_weights = os.path.join("trains_record", "Congestion_game_with_Adam", "Simple.pth")
input_size = 1 * K
output_size = 1 * K
model = Shallow_Congestion(input_size=input_size, output_size=output_size)  # Initialize the model with the same architecture
model.load_state_dict(torch.load(file_path_weights, map_location='cpu'))
# Take beta results
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    X_test = torch.tensor(X_test, dtype=torch.float32)
    # Squeeze the input tensor to match the Fc size
    # X_test = X_test.squeeze(dim=-1)
    outputs = model(X_test)
    # outputs = outputs.squeeze(dim=-1)
    outputs = outputs.numpy()

# Generate parameters to calculate beta
# A_NN = outputs[:, 0:K]
# B_NN = outputs[:, 0:K]
# C_NN = outputs[:, K:(2 * K)]
C_k_est = outputs[:, 0:K]


# Calculate some evaluation metrics (you can customize this part)
# mse_C = ((C_NN - C)**2).mean()
# mse_B = ((B_NN - B)**2).mean()
# mse_A = ((A_NN - A)**2).mean()
print(f"L = {L}")
print(f"K = {K}")
# print("Mean Squared Error C:", mse_C)
# print("Mean Squared Error B:", mse_B)
# print("Mean Squared Error A:", mse_A)
mse_Ck = ((C_k_est - Ck) ** 2).mean()
print("Mean Squared Error Ck:", mse_Ck)


# Plot Results
# Visualize the results with different colors for actual and predicted values
# plt.figure(1)
# plt.figure(figsize=(10, 6))
# plt.scatter(B.flatten(), B_NN.flatten(), c='blue', label='Predict Values', alpha=0.5)
# plt.scatter(B.flatten(), B.flatten(), c='red', label='Actual Values', alpha=0.5)
# plt.xlabel("Actual Values (B)")
# plt.ylabel("Predicted Values (B)")
# plt.title("Actual vs. Predicted Values (B)")
# plt.legend()

plt.figure(1)
plt.figure(figsize=(10, 6))
plt.scatter(Ck.flatten(), C_k_est.flatten(), c='blue', label='Predict Values', alpha=0.5)
plt.scatter(Ck.flatten(), Ck.flatten(), c='red', label='Actual Values', alpha=0.5)
plt.xlabel("Actual Values (Ck)")
plt.ylabel("Predicted Values (Ck)")
plt.title("Actual vs. Predicted Values (Ck)")
plt.legend()
plt.show()

# plt.figure(2)
# plt.figure(figsize=(10, 6))
# plt.scatter(C.flatten(), C_NN.flatten(), c='blue', label='Predict Values', alpha=0.5)
# plt.scatter(C.flatten(), C.flatten(), c='red', label='Actual Values', alpha=0.5)
# plt.xlabel("Actual Values (C)")
# plt.ylabel("Predicted Values (C)")
# plt.title("Actual vs. Predicted Values (C)")
# plt.legend()


# plt.figure(3)
# plt.figure(figsize=(10, 6))
# plt.scatter(A.flatten(), A_NN.flatten(), c='blue', label='Predict Values', alpha=0.5)
# plt.scatter(A.flatten(), A.flatten(), c='red', label='Actual Values', alpha=0.5)
# plt.xlabel("Actual Values (A)")
# plt.ylabel("Predicted Values (A)")
# plt.title("Actual vs. Predicted Values (A)")
# plt.legend()

# plt.show()
print("Finsh ....")