# -*- coding: utf-8 -*-
"""
Created on : ------

@author: Ariel_Kantorovich
"""
import matplotlib.pyplot as plt
import numpy as np
from numba import jit

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
    B = np.random.uniform(low=-beta, high=beta, size=(L, N, 1))
    # Make Q to Q.T@Q because we want convex prolem with optimum solution
    Q = np.transpose(Q, (0, 2, 1)) @ Q
    return Q, B

@jit(nopython=True)
def calculate_scores(Q, B, x_optimal, L):
    """
    :param Q: size (L,NxN)
    :param B: size (L, Nx1)
    :param x_optimal: size (L, Nx1)
    :param L: scalar
    :return: scores vector by size L
    """
    scores = np.zeros((L, ))
    for i in range(L):
        x = x_optimal[i]
        scores[i] = np.sum(0.5 * np.transpose(x) @ Q[i] @ x + np.transpose(x) @ B[i])
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

@jit(nopython=True)
def compute_Nash_Vec(Qn, Bn):
    """
    :param Qn: size (L, N, NxN)
    :param Bn: size (L, N, Nx1)
    :return: x_nash size (L, N, 1)
    """
    L, N, _, _ = Qn.shape
    x_nash = np.zeros((L, N, 1))
    for l in range(L):
        Q_row = np.zeros((N, N))
        Q_col = np.zeros((N, N))
        B_new = np.zeros((N, 1))
        for n in range(N):
            Q_row[n, :] = Qn[l, n, n, :]
            Q_col[n, :] = Qn[l, n, :, n]
            B_new[n] = Bn[l, n, n]
        Q_psudo_inv = np.linalg.pinv(Q_row + Q_col)
        x_nash[l] = -2 * Q_psudo_inv @ B_new
    return x_nash



# %% Parameters
L = 600 # samples of Q
N = 5 # Number of players
alpha = 4.0
beta = 0.5
std_list = np.arange(0.0, 15.3, 0.1)
mu = 0.0
mu_B = 0.0
std_B = 0.01
add_diag = False
Border_projection = 2
# Define final results arrays
cost_optimal_list = np.zeros_like(std_list)
cost_Q_noise_list = np.zeros_like(std_list)
cost_X_noise_list = np.zeros_like(std_list)
cost_X_fake_list = np.zeros_like(std_list)

cost_X_optimal_norm_list = np.zeros_like(std_list)
cost_Q_noise_norm_list = np.zeros_like(std_list)
cost_X_noise_norm_list = np.zeros_like(std_list)
cost_X_fake_norm_list = np.zeros_like(std_list)
# Generate Q and Q_noise
Q, B = generate_Q_B(N, L, alpha, beta, subMean=False)
# Add diagonal from inverse Matrix
if add_diag:
    c_diag = 10.0
    identity = c_diag * np.eye(N)
    Q = Q + identity[np.newaxis, :, :]
# Repeat for N players
B_repeat = np.repeat(B[:, np.newaxis, :, :], N, axis=1)
Q_repeat = np.repeat(Q[:, np.newaxis, :, :], N, axis=1)
# Inverse Matrices
Q_inv = np.linalg.inv(Q)
for i, std in enumerate(std_list):
    # Generate Q, B noise
    Q_noise = Q_repeat + std * np.random.randn(L, N, N, N) + mu
    B_noise = B_repeat + std_B * np.random.randn(L, N, N, 1) + mu_B
    # Generate Q_fake for debug
    temp = np.mean(Q_noise, axis=(2, 3))
    Q_fake = temp[:, :, np.newaxis, np.newaxis] * np.ones((N, N))
    # Calculate X action of all three methods
    x_opt = np.matmul(-Q_inv, B)
    x_Noise = x_opt + std * np.random.randn(L, N, 1) + mu
    x_QNoise = compute_Nash_Vec(Q_noise, B_noise)  # calculate nash
    x_fake = compute_Nash_Vec(Q_fake, B_noise)
    # Project solution to solution space
    x_opt = np.minimum(np.maximum(x_opt, -Border_projection), Border_projection)
    x_QNoise = np.minimum(np.maximum(x_QNoise, -Border_projection), Border_projection)
    x_Noise = np.minimum(np.maximum(x_Noise, -Border_projection), Border_projection)
    x_fake = np.minimum(np.maximum(x_fake, -Border_projection), Border_projection)
    # Calculate cost
    cost_optimal = calculate_scores(Q, B, x_opt, L)
    cost_Q_Noise = calculate_scores(Q, B, x_QNoise, L)
    cost_X_noise = calculate_scores(Q, B, x_Noise, L)
    cost_X_fake = calculate_scores(Q, B, x_fake, L)
    # Sum on L games
    cost_optimal_list[i] = np.sum(cost_optimal)
    cost_Q_noise_list[i] = np.sum(cost_Q_Noise)
    cost_X_noise_list[i] = np.sum(cost_X_noise)
    cost_X_fake_list[i] = np.sum(cost_X_fake)
    # calculate norm to each action
    x_opt_norm = np.sum(np.linalg.norm(x_opt, axis=1)) * (1 / L)
    x_QNoise_norm = np.sum(np.linalg.norm(x_QNoise, axis=1)) * (1 / L)
    x_Noise_norm = np.sum(np.linalg.norm(x_Noise, axis=1)) * (1 / L)
    x_fake_norm = np.sum(np.linalg.norm(x_fake, axis=1)) * (1 / L)
    # Put in Norm list
    cost_X_optimal_norm_list[i] = x_opt_norm
    cost_X_fake_norm_list[i] = x_fake_norm
    cost_Q_noise_norm_list[i] = x_QNoise_norm
    cost_X_noise_norm_list[i] = x_Noise_norm


# Plot results
plt.figure(1)
plt.plot(std_list, cost_optimal_list, '--k', label='Opt')
plt.plot(std_list, cost_Q_noise_list, label='Q Noise')
plt.plot(std_list, cost_X_noise_list, label='X Noise')
plt.plot(std_list, cost_X_fake_list, label='Q fake(Mean)')
plt.legend(), plt.ylabel("# Cost"), plt.xlabel("$\sigma$"), \

plt.figure(2)
plt.plot(std_list, cost_X_optimal_norm_list, '--k', label='||Opt||')
plt.plot(std_list, cost_Q_noise_norm_list, label='||Q Noise||')
plt.plot(std_list, cost_X_noise_norm_list, label='||X Noise||')
plt.plot(std_list, cost_X_fake_norm_list, label='||Q fake(Mean)||')
plt.legend(), plt.ylabel("# Cost"), plt.xlabel("$\sigma$"), \

plt.show()

print("Finsh !!! !")