# -*- coding: utf-8 -*-
"""
Created on : ------
@author: Ariel_Kantorovich
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit


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


def generate_gain_channel(L, N, alpha):
    """
    :param L: (int) trials of game
    :param N:  (int) number of players
    :param alpha:  (int) some constant
    :return: g return the gain of channels between players
    """
    # Generate random transceiver locations in a radius of 1
    Transreceivers = np.random.rand(L, N, 2) * 2 - 1  # Scale to (-1, 1) and then to (-1, 1) radius 1
    # Generate random receiver locations in a radius of 0.1 around each transceiver
    Rlink = 0.1
    ReceiverOffsets = Rlink * (np.random.rand(L, N, 2) * 2 - 1)  # Scale to (-1, 1) and then to (-0.1, 0.1)
    Receivers = Transreceivers + ReceiverOffsets
    # Calculate distances between transceivers and receivers
    distances = np.linalg.norm(Transreceivers[:, :, np.newaxis, :] - Receivers[:, np.newaxis, :, :], axis=3)
    g = alpha / (distances ** 2)
    return g

def multi_wireless_loop(N, L, T, g, lr, P, beta):
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

    # Finally Let's return last P and global fot L trials
    return P_record[T-1].squeeze(), global_objective[T-1]

# Parameters:
N = 5
alpha = 10e-3
L = 2000
T = 40000
learning_rate = 0.03 * np.reciprocal(np.power(range(1, T + 1), 0.65))
# learning_rate = 0.0001 * np.ones((T, ))
add_gain = False
add_gain_param = 1000.0

# Generate gain matrix
g = generate_gain_channel(L, N, alpha)
# Add Gain to transmiter channel
if add_gain:
    g_channel = add_gain_param * np.eye(N)
    g = g + g_channel

# grid Search Algorithm that find the best beta
beta_save = np.zeros((L,))
beta_list = np.linspace(0.3, 1.0, 25)
best_global = np.zeros((L, ))

# Build X_train
P_init = 0.1 * np.random.rand(L, N, 1)  # Generate Power from uniform distributed
g_square = g ** 2
g_diag = np.diagonal(g_square, axis1=1, axis2=2)
g_diag = g_diag.reshape(L, N, 1)
g_colum = np.transpose(g_square, axes=(0, 2, 1))
In_init = np.matmul(g_colum, P_init) - g_diag * P_init
X_train = np.concatenate([P_init, In_init, g_diag], axis=1)
X_train = X_train.squeeze()

# Save X_train
np.save("Numpy_array_save/X_train.npy", X_train)

# Initialize state with beta = 0
P_n, global_n = multi_wireless_loop(N, L, T, g, learning_rate, P_init, beta=0)
best_global = global_n
for beta in beta_list:
    P_n, global_n = multi_wireless_loop(N, L, T, g, learning_rate, P_init, beta)
    # Update beta
    condition = global_n > best_global
    beta_save[condition] = beta
    best_global[condition] = global_n[condition]

# Save Y_train
np.save("Numpy_array_save/Y_train.npy", beta_save)

print("Finsh !!!")