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

# %% Setting Functions
def calculate_scores(Q, B, x_optimal, N_exp, N, L):
    """
    :param Q: size (L, N, N)
    :param B: size (L, N, 1)
    :param x_optimal: size (Nexp, N)
    :param N_exp: scalar
    :param N: scalar
    :param L: scalar
    :return: scores vector by size N_exp
    """
    result_pass = 0
    for i in range(N):
        result_pass += 0.5 * x_optimal[i].T @ Q @ x_optimal[i] + x_optimal[i].T @ B
    result_pass /= N
    return result_pass

def compute_gradient(Qn, Bn, x):
    """
    Compute the gradient of the cost function for each agent, Pay attention our assumption is that Q is symmetric
    :param Qn: size dxd
    :param Bn: size dx1
    :param x: size dx1
    :return: derivative size dx1
    """
    return np.matmul(Qn, x) + Bn

def multi_agent_gradient_descent(N, T, learning_rate,
                                 Qn_mean, Qn_NN, Bn, recordFlag, Q, B, L, Border=1):
    """
    :param Q, B where Q is prob distance map size (L, N, N) and B is one vector size (L, N, 1)
    :param recordFlag: bool track the progress of cost and agent
    :param N_exper: int, number of trials for Qn Bn
    :param N: int, number of players
    :param T: int, iteration time
    :param learning_rate: size (T, )
    :param Qn: size (N_expr, N, N)
    :param Bn: size (N_expr, N, 1)
    :return: x_ agents size (N_expr, N)
             x_record size (T, N_expr, N)
             cost_record size (T, N_expr)
    """
    # Init learning rate
    lr2 = 0.001 * np.ones((T,))  # mean
    lr2[19000:] = 0.0001
    lr1 = 0.001 * np.ones((T, )) # mean
    lr1[9000:] = 0.0001
    Qn_Randomize = np.random.rand(N, N, N)
    # Initialize x_agents
    x_agents_mean = 0.01 * np.ones((N, N, 1))
    x_agents_NN = 0.01 * np.ones((N, N, 1))
    x_agents_Randomiz = 0.01 * np.ones((N, N, 1))
    # Initialize x_agents and cost to track
    x_record_mean = np.zeros((T, N, N, 1))
    x_record_NN = np.zeros((T, N, N, 1))
    x_record_Randomize = np.zeros((T, N, N, 1))
    cost_record_mean = np.zeros((T, ))
    cost_record_NN = np.zeros((T,))
    cost_record_Randomize = np.zeros((T,))
    # loop over time
    for t in range(T):
        # Compute gradients in parallel
        gradients_means = compute_gradient(Qn_mean, Bn, x_agents_mean)
        gradients_NN = compute_gradient(Qn_NN, Bn, x_agents_NN)
        gradients_Ranomize = compute_gradient(Qn_Randomize, Bn, x_agents_Randomiz)
        # Update the agent's variable 'x' using gradient descent
        x_agents_mean -= lr1[t] * gradients_means
        x_agents_NN -= lr2[t] * gradients_NN
        x_agents_Randomiz -= learning_rate[t] * gradients_Ranomize
        # Project the action to [-1,1] (Normalization)
        x_agents_mean = np.minimum(np.maximum(x_agents_mean, -Border), Border)
        x_agents_NN = np.minimum(np.maximum(x_agents_NN, -Border), Border)
        x_agents_Randomiz = np.minimum(np.maximum(x_agents_Randomiz, -Border), Border)
        if recordFlag:
            x_record_mean[t] = x_agents_mean
            x_record_NN[t] = x_agents_NN
            x_record_Randomize[t] = x_agents_Randomiz
            cost_record_mean[t] = calculate_scores(Q, B, x_agents_mean, N, N, L)
            cost_record_NN[t] = calculate_scores(Q, B, x_agents_NN, N, N, L)
            cost_record_Randomize[t] = calculate_scores(Q, B, x_agents_Randomiz, N, N, L)
    return x_agents_mean, x_agents_NN, x_agents_Randomiz, x_record_mean, x_record_NN, x_record_Randomize, cost_record_mean, cost_record_NN, cost_record_Randomize


# %% Parameters
L = 100 # samples of Q
input_size = 5
N = 5 # Number of players
output_size = N ** 2  # Size of output (vectorized Q matrix)
recordFlag = True # track the progress of cost and agent
T = 35000 # Number of iteration
learning_rate = 0.05 * np.reciprocal(np.power(range(1, T + 1), 0.7))
Bn = np.ones((N, N, 1))
B = np.ones((N, 1))
sample = 0 # sample from test to check
Border_projection = 35
# %% Main loop
# Step 1: mean of Q_train this will be new Qn_mean
file_path_Q = os.path.join("Numpy_array_save", "train_data_set(Xsize5)", "Q.npy")
Q_train = np.load(file_path_Q)
Qn_mean = np.mean(Q_train, axis=0)
Qn_mean = np.expand_dims(Qn_mean, axis=0).repeat(N, axis=0)

# Step 2: upload test set ang generate Qn_NN using NN
file_path_x = os.path.join("Numpy_array_save", "test_data_set(Xsize5)", "x_test.npy")
file_path_y = os.path.join("Numpy_array_save", "test_data_set(Xsize5)", "y_test.npy")
file_path_Q = os.path.join("Numpy_array_save", "test_data_set(Xsize5)", "Q.npy")
X_test = np.load(file_path_x)
Y_test = np.load(file_path_y)
Q_test = np.load(file_path_Q)

file_path_weights = os.path.join("trains_record", "train_2000epochs_SGD(X=5)", "Q_Net.pth")
model = QNetwork(input_size, output_size)  # Initialize the model with the same architecture
model.load_state_dict(torch.load(file_path_weights))
model.eval()  # Set the model to evaluation mode

with torch.no_grad():
    # Sample example from test and squezze
    X_sample = torch.tensor(X_test[sample*N:(sample + 1)*N], dtype=torch.float32)
    X_sample = X_sample.squeeze(dim=-1)
    outputs = model(X_sample)
    predicted_values = outputs.numpy()

Qn_NN = predicted_values.reshape(-1, N, N)
Q = Q_test[sample] # Sample Q from test set

# Step 3: Run optimization on two games parallel
x_agents_mean, x_agents_NN, x_agents_Randomiz, x_record_mean, x_record_NN, x_record_Randomize, cost_record_mean, cost_record_NN, cost_record_Randomize = multi_agent_gradient_descent(N, T, learning_rate,
                                                                                                                        Qn_mean, Qn_NN, Bn, recordFlag, Q, B, L, Border=Border_projection)

# %% Plot results
t = np.arange(T)
plt.plot(t, cost_record_mean, label="Mean"),
plt.plot(t, cost_record_NN, label="NN"),
plt.plot(t, cost_record_Randomize, label="Randomize"),
plt.xlabel("# Iteration"), plt.ylabel("candadite_score"), plt.legend()
plt.show()
print("Finsh")



