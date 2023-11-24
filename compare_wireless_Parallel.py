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
from numba import jit
import scipy.optimize as optim
from joblib import Parallel, delayed

# Define seed for check
np.random.seed(0)
# %% Setting Archticture network
class Wireless_AutoEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Wireless_AutoEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 128)
        self.bn4 = nn.BatchNorm1d(128)
        # Define Decoder
        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.fc6 = nn.Linear(64, 32)
        self.bn6 = nn.BatchNorm1d(32)
        self.fc7 = nn.Linear(32, 16)
        self.bn7 = nn.BatchNorm1d(16)
        self.fc8 = nn.Linear(16, output_size)
        self.init_weights()

    def forward(self, x):
        # Encoder Part
        x1 = torch.relu(self.bn1(self.fc1(x)))
        x2 = torch.relu(self.bn2(self.fc2(x1)))
        x3 = torch.relu(self.bn3(self.fc3(x2)))
        x4 = torch.relu(self.bn4(self.fc4(x3)))
        # Decoder Part
        x5 = torch.relu(self.bn5(self.fc5(x4))) + x3
        x6 = torch.relu(self.bn6(self.fc6(x5))) + x2
        x7 = torch.relu(self.bn7(self.fc7(x6))) + x1
        # Final layer without activation for regression
        output = torch.relu(self.fc8(x7))
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


@jit(nopython=True)
def calc_second_gradient(N, gradients_second, g_diag, g_square, P, In):
    """
    :param N: int number of Players
    :param gradients_second: zero numpy array size (L, N, 1)
    :param g_diag: numpy array size (L, N, 1)
    :param g_square: numpy array size (L, N, N)
    :param P: numpy array size (L, N, 1)
    :param In: numpy array size (L, N, 1)
    :return: gradients_second: numpy array size (L, N, 1)
    """
    N0 = 0.001
    L, _, _ = gradients_second.shape
    for n in range(N):
        for j in range(N):
            if n != j:
                for l in range(L):
                    gradients_second[l, n, 0] += (
                            g_diag[l, j, 0] * g_square[l, n, j] * P[l, j, 0] /
                            ((In[l, j, 0] + N0) * (In[l, j, 0] + N0 + g_diag[l, j, 0] * P[l, j, 0]))
                    )
    return -1.0 * gradients_second


def generate_gain_channel(L, N, alpha, disck=1):
    """
    :param L: (int) trials of game
    :param N:  (int) number of players
    :param alpha:  (int) some constant
    :return: g return the gain of channels between players
    """
    # Generate random transceiver locations in a radius of 1
    Transreceivers = np.random.rand(L, N, 2) * 2 - disck  # Scale to (-1, 1) and then to (-1, 1) radius 1
    # Generate random receiver locations in a radius of 0.1 around each transceiver
    Rlink = 0.1
    ReceiverOffsets = Rlink * (np.random.rand(L, N, 2) * 2 - disck)  # Scale to (-1, 1) and then to (-0.1, 0.1)
    Receivers = Transreceivers + ReceiverOffsets
    # Calculate distances between transceivers and receivers
    distances = np.linalg.norm(Transreceivers[:, :, np.newaxis, :] - Receivers[:, np.newaxis, :, :], axis=3)
    g = alpha / (distances ** 2)
    return g

def multi_wireless_loop(N, L, T, g, lr, P, beta=0):
    """
    :param N: (int) number of players
    :param L: (int) trials of game
    :param T: (int) duration of experiment
    :param g: (L, N, N) channel gain matrix
    :param lr: (T, ) learning rate vector
    :return: mean_P_record (T, N) the P array with record on time
    """
    # Initialize Parameters
    Border_floor = 0
    Border_ceil = 1
    N0 = 0.001
    g_square = g ** 2
    # Initialize record variables
    P_record = np.zeros((T, L, N, 1))
    global_objective = np.zeros((T, L))
    gradients_record = np.zeros((T, L, N, 1))
    # Prepare g to calculation
    g_diag = np.diagonal(g_square, axis1=1, axis2=2)
    g_diag = g_diag.reshape(L, N, 1)
    g_colum = np.transpose(g_square, axes=(0, 2, 1))
    # Initialize gradients array
    gradients_first = np.zeros((L, N, 1))
    gradients_second = np.zeros((L, N, 1))
    for t in range(T):
        # calculate instance
        In = np.matmul(g_colum, P) - g_diag * P

        # calculate gradients
        numerator = (g_diag / (In + N0))
        gradients_first = (numerator / (1 + numerator * P)) - beta
        gradients_second = calc_second_gradient(N, gradients_second, g_diag, g_square, P, In)

        # update agent vector(P)
        P += lr[t] * (gradients_first + gradients_second)
        # Project the action to [Border_floor, Border_ceil] (Normalization)
        P = np.minimum(np.maximum(P, Border_floor), Border_ceil)

        # Save results in record
        P_record[t] = P
        gradients_record[t] = gradients_first + gradients_second

        # Calculate global objective
        temp = np.log(1 + numerator * P)
        temp = temp.squeeze()
        global_objective[t] = np.sum(temp, axis=1)

    # Finally Let's mean for all L trials
    P_record = P_record.squeeze()
    gradients_record = gradients_record.squeeze()
    mean_P_record = np.mean(P_record, axis=1)
    mean_gradients_record = np.mean(gradients_record, axis=1)
    mean_global_objective = np.sum(global_objective, axis=1)
    return mean_P_record, mean_global_objective

def objective_function(P, *args):
    """
    We define the sum with minus because in general scipy optimization algorithms try to minimize objective
    :param P:
    :param g:
    :param I:
    :param N0:
    :return:
    """
    g_diag, g_colum, N0 = args
    P = P.reshape(P.shape[0], 1)
    I = np.matmul(g_colum, P) - g_diag * P
    numerator = ((g_diag * P) / (I + N0))
    x = np.sum(np.log(1 + numerator))
    return -1 * x

def optimize_trial(g_diag, g_colum, N0, bounds):
    result = optim.differential_evolution(objective_function, bounds, args=(g_diag, g_colum, N0), maxiter=10000)
    optimal_power_allocation = result.x
    optimal_objective_value = -result.fun
    return optimal_power_allocation, optimal_objective_value

def Calculate_Beta_NN(L, N, g_colum, g_diag, AutoEncoder=False):
    # Build Input
    X_test = np.zeros((L * N, 4))
    In = np.matmul(g_colum, np.ones((L, N, 1))) - g_diag * np.ones((L, N, 1))
    X_test[:, 0] = np.log(g_diag[:, :, 0]).flatten()
    X_test[:, 1] = np.log(In[:, :, 0]).flatten()
    X_test[:, 2] = np.log(1 + (g_diag[:, :, 0] / (In[:, :, 0] + N0))).flatten()
    if AutoEncoder:
        kernel = np.exp(- np.abs(np.log(In[:, :, 0]) - np.log(g_diag[:, :, 0])) / 2).flatten()
    else:
        kernel = np.exp(- np.abs(In[:, :, 0] - g_diag[:, :, 0]) / 2).flatten()

    X_test[:, 3] = kernel

    # Take beta results
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        X_test = torch.tensor(X_test, dtype=torch.float32)
        # Squeeze the input tensor to match the Fc size
        X_test = X_test.squeeze(dim=-1)
        outputs = model(X_test)
        outputs = outputs.squeeze(dim=-1)
        beta = outputs.numpy()
    beta = beta.reshape(L, N, 1)
    return beta




######################################## Main Program #############################################################################

# Initialize HyperParameters
N0 = 0.001
L = 100
T = 40000
learning_rate_1 = 0.03 * np.reciprocal(np.power(range(1, T + 1), 0.65))
learning_rate_2 = 0.03 * np.reciprocal(np.power(range(1, T + 1), 0.9))
N = 5
alpha = 10e-2
input_size = 4
output_size = 1

# Initialize the model with the same architecture
file_path_weights = os.path.join("trains_record", "Wireless_example", "Wireless.pth")
model = Wireless(input_size, output_size)
model.load_state_dict(torch.load(file_path_weights, map_location='cpu'))

# Generate gains and initialize power allocations
g = generate_gain_channel(L, N, alpha, disck=1)
g_square = g ** 2
P = 0.01 * np.random.rand(L, N, 1)

# Prepare g for calculation
g_diag = np.diagonal(g_square, axis1=1, axis2=2)
g_diag = g_diag.reshape(L, N, 1)
g_colum = np.transpose(g_square, axes=(0, 2, 1))

# Calculate beta from Neural Network
beta_NN = Calculate_Beta_NN(L, N, g_colum, g_diag, AutoEncoder=False)

# Define bounds for P, where each P_n should be between 0 and 1 (Optimization constraint)
bounds = [(0, 1)] * N

# Parallel optimization using joblib
results = Parallel(n_jobs=-1)(
    delayed(optimize_trial)(g_diag[i], g_colum[i], N0, bounds) for i in range(L)
)

# Unpack the results
optimal_power_allocations, optimal_objective_values = zip(*results)
# Convert tuple to array
optimal_power_allocations = np.array(optimal_power_allocations)
optimal_objective_values = np.array(optimal_objective_values)
# Calculate Mean of optimal results
global_optim = np.sum(optimal_objective_values)
P_optimal = np.mean(optimal_power_allocations, axis=0)

# calculate results for Naive approach
beta_naive = 0
P_naive_mean, global_naive_mean = multi_wireless_loop(N, L, T, g, learning_rate_1, P, beta=beta_naive)
# calculate results for constant mean search approach
beta_search = 0.7925
P_search_mean, global_search_mean = multi_wireless_loop(N, L, T, g, learning_rate_1, P, beta=beta_search)
# Calculate for NN
P_net_mean, global_net_mean = multi_wireless_loop(N, L, T, g, learning_rate_1, P, beta=beta_NN)

# Calculate percentage differences
optim_prec_diff = 0.0
naiv_prec_diff = (np.abs(global_naive_mean[T-1] - global_optim) / global_optim) * 100
net_prec_diff = (np.abs(global_net_mean[T-1] - global_optim) / global_optim) * 100
search_prec_diff = (np.abs(global_search_mean[T-1] - global_optim) / global_optim) * 100

# Plot results
t = np.arange(T)
plt.figure(1)
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
for n in range(N):
    Pn = P_naive_mean[:, n]
    plt.plot(t, Pn, label=f"P{n}"),
    plt.ylabel("# candidate action"), plt.legend()
    plt.title("Naive")
plt.subplot(2, 2, 2)
for n in range(N):
    Pn = P_net_mean[:, n]
    plt.plot(t, Pn, label=f"P{n}"),
    plt.ylabel("# candidate action"), plt.legend()
    plt.title("Network")
plt.subplot(2, 2, 3)
for n in range(N):
    Pn = np.ones((T, )) * P_optimal[n]
    plt.plot(t, Pn, label=f"P{n}"), plt.xlabel("# Iteration"),
    plt.ylabel("# candidate action"), plt.legend()
    plt.title("Optimization Package solution")
plt.subplot(2, 2, 4)
for n in range(N):
    Pn = P_search_mean[:, n]
    plt.plot(t, Pn, label=f"P{n}"), plt.xlabel("# Iteration"),
    plt.ylabel("# candidate action"), plt.legend()
    plt.title(fr"Mean $\beta$, $\beta$={beta_search}")

plt.figure(2)
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.title("Compare:Naive,Beta-Network, Mean-Beta")
plt.plot(t, global_naive_mean, label='naive'),
plt.plot(t, global_net_mean, label='NN'),
plt.plot(t, global_search_mean, label='Mean'),
plt.plot(t, np.ones(T, ) * global_optim, '--k', label='Optimal package')
plt.xlabel("# Iteration"), plt.ylabel("Global Objective"), plt.legend()
plt.subplot(1, 2, 2)
plt.title("Zoom-in")
plt.plot(t, global_naive_mean, label='naive'),
plt.plot(t, global_net_mean, label='NN'),
plt.plot(t, global_search_mean, label='Mean'),
plt.plot(t, np.ones(T, ) * global_optim, '--k', label='Optimal package')
plt.xlabel("# Iteration"), plt.ylabel("Global Objective"), plt.legend()
# Set y-axis range
plt.ylim(3500, 4100)


plt.figure(3)
plt.figure(figsize=(10, 6))
plt.title("Total Power")
plt.plot(t, np.ones(T, ) * np.sum(P_naive_mean[T-1, :]), label='naive'),
plt.plot(t, np.ones(T, ) * np.sum(P_net_mean[T-1, :]), label='NN'),
plt.plot(t, np.ones(T, ) * np.sum(P_search_mean[T-1, :]), label='Mean'),
plt.plot(t, np.ones(T, ) * np.sum(P_optimal), '--k', label='Optimal package'),
plt.xlabel("# Iteration"), plt.legend()


plt.figure(4)
plt.figure(figsize=(10, 6))
plt.title("Global Percentage Difference")
plt.plot(t, np.ones(T, ) * optim_prec_diff, '--k', label='Optimal package'),
plt.plot(t, np.ones(T, ) * naiv_prec_diff, label='naive'),
plt.plot(t, np.ones(T, ) * net_prec_diff, label='NN'),
plt.plot(t, np.ones(T, ) * search_prec_diff, label='Mean')
plt.xlabel("# Iteration"), plt.legend()
plt.show()


print("Finsh !!!")


