# -*- coding: utf-8 -*-
"""
Created on : ------

@author: Ariel_Kantorovich
"""
import numpy as np
import matplotlib.pyplot as plt

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

def multi_wireless_loop(N, L, T, g, lr):
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
    P = np.random.rand(L, N, 1) # Generate Power from uniform distributed
    # Initialize record variables
    P_record = np.zeros((T, L, N, 1))
    # Prepare g to calculation
    g_diag = np.diagonal(g_square, axis1=1, axis2=2)
    g_diag = g_diag.reshape(L, N, 1)
    g_colum = np.transpose(g_square, axes=(0, 2, 1))
    for t in range(T):
        # calculate instance
        In = np.matmul(g_colum, P)
        # calculate gradients
        numerator = (g_diag / (In + N0))
        gradients = numerator / (1 + numerator * P)
        # update agent vector(P)
        P -= lr[t] * gradients
        # Project the action to [Border_floor, Border_ceil] (Normalization)
        P = np.minimum(np.maximum(P, Border_floor), Border_ceil)
        # Save results in record
        P_record[t] = P
    # Finaly Let's mean for all L trials
    P_record = P_record.squeeze()
    mean_P_record = np.mean(P_record, axis=1)
    return mean_P_record

# Parameters:
N = 5
alpha = 10e-3
L = 300
T = 35000
learning_rate = 0.009 * np.reciprocal(np.power(range(1, T + 1), 0.65))
# Generate gain matrix
g = generate_gain_channel(L, N, alpha)
final_cost = multi_wireless_loop(N, L, T, g, learning_rate)
# Plot results
t = np.arange(T)
for n in range(N):
    Pn = final_cost[:, n]
    plt.plot(t, Pn, label=f"P{n}"), plt.xlabel("# Iteration"),
    plt.ylabel("# candidate action"), plt.legend()
plt.show()

print("Finsh !!!")