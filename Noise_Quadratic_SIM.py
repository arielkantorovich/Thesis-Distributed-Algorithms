# -*- coding: utf-8 -*-
"""
Created on : ------

@author: Ariel_Kantorovich
"""
import matplotlib.pyplot as plt
import numpy as np

def generate_Q_B(N, L, alpha, beta, std_Q=0, mu_Q=0, subMean=False):
    """
    :param SubMean: boolean variable if subtract mean from Q or not
    :param std_Q: variance for Q scalar
    :param mu_Q: mu for Q scalar
    :param N: Number of players
    :param L: Trials to experiment
    :param alpha: scalar
    :param beta:  scalar
    :return: Q, B, Q_noise  where Q is prob distance map size (L, N, N) and B is one vector size (L, N, 1) and Q_noise is Q with addative Gaussian Noise
    """
    # Step 1: Generate L different sets of N random points
    points = np.random.rand(L, N, 2)
    # Step 2: Build distance maps D for all trials
    points_expanded = points[:, np.newaxis, :, :]
    differences = points_expanded - points[:, :, np.newaxis, :]
    distances = np.linalg.norm(differences, axis=-1)
    # Step 3: Generate Q = exp(-alpha*D) for all trials
    Q = np.exp(-alpha * distances)
    if subMean:
        # In this part we subtract the mean from every L trial in Q
        trial_means = np.mean(Q, axis=(1, 2))
        # Expand trial_means to have the same shape as Q (L, N, N)
        trial_means_expanded = trial_means[:, np.newaxis, np.newaxis]
        Q = Q - trial_means_expanded
    # Generate B of size Nx1xL
    B = beta * np.ones((L, N, 1))
    # Make Q to Q.T@Q because we want convex prolem with optimum solution
    Q = np.transpose(Q, (0, 2, 1)) @ Q
    return Q, B

def calculate_scores(Q, B, x_optimal, L):
    """
    :param Q: size (L,N, N, N)
    :param B: size (L, N, N, 1)
    :param x_optimal: size (L, N, N, 1)
    :param N_exp: scalar
    :param N: scalar
    :param L: scalar
    :return: scores vector by size L
    """
    scores = np.zeros((L, ))
    for i in range(L):
        x = x_optimal[i]
        x_transposed = np.transpose(x, (0, 2, 1))
        scores[i] = np.sum(0.5 * x_transposed @ Q[i] @ x + x_transposed @ B[i])
    return scores

def compute_gradient(Qn, Bn, x):
    """
    Compute the gradient of the cost function for each agent, Pay attention our assumption is that Q is symmetric
    :param Qn: size dxd
    :param Bn: size dx1
    :param x: size dx1
    :return: derivative size dx1
    """
    return np.matmul(Qn, x) + Bn


# %% Parameters
L = 100 # samples of Q
N = 5 # Number of players
alpha = 4.0
beta = 0.1
std_list = np.arange(0.1, 10.3, 0.1)
mu = 0
add_diag = False
Border_projection = 70
# Define final results arrays
cost_optimal_list = np.zeros_like(std_list)
cost_Q_noise_list = np.zeros_like(std_list)
cost_X_noise_list = np.zeros_like(std_list)
# Generate Q and Q_noise
Q, B = generate_Q_B(N, L, alpha, beta, subMean=False)
# Add diagonal from inverse Matrix
if add_diag:
    c_diag = 10.0
    identity = c_diag * np.eye(N)
    Q = Q + identity[np.newaxis, :, :]
# Repeat for N players
B = np.repeat(B[:, np.newaxis, :, :], N, axis=1)
Q = np.repeat(Q[:, np.newaxis, :, :], N, axis=1)
# Inverse Matrices
Q_inv = np.linalg.inv(Q)
for i, std in enumerate(std_list):
    # Generate Q noise
    Q_noise = Q + std * np.random.randn(L, N, N, N) + mu
    # Inverse Matrices
    Q_inv_noise = np.linalg.inv(Q_noise)
    # Calculate X action of all three methods
    x_opt = np.matmul(-Q_inv, B)
    x_QNoise = np.matmul(-Q_inv_noise, B)
    x_Noise = x_opt + std * np.random.randn(L, N, N, 1) + mu
    # Project solution to solution space
    x_opt = np.minimum(np.maximum(x_opt, -Border_projection), Border_projection)
    x_QNoise = np.minimum(np.maximum(x_QNoise, -Border_projection), Border_projection)
    x_Noise = np.minimum(np.maximum(x_Noise, -Border_projection), Border_projection)
    # Calculate cost
    cost_optimal = calculate_scores(Q, B, x_opt, L)
    cost_Q_Noise = calculate_scores(Q, B, x_QNoise, L)
    cost_X_noise = calculate_scores(Q, B, x_Noise, L)
    # Sum on L games
    cost_optimal_list[i] = np.sum(cost_optimal)
    cost_Q_noise_list[i] = np.sum(cost_Q_Noise)
    cost_X_noise_list[i] = np.sum(cost_X_noise)


# Plot results
plt.plot(std_list, cost_optimal_list, '--k', label='Opt')
plt.plot(std_list, cost_Q_noise_list, label='Q Noise')
plt.plot(std_list, cost_X_noise_list, label='X Noise')
plt.legend(), plt.ylabel("# Cost"), plt.xlabel("$\sigma$"), plt.show()

print("Finsh !!! !")