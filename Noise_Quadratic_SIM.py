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
    # Generate Q noise
    Q_noise = Q + std_Q * np.random.randn(L, N, N) + mu_Q
    return Q, B, Q_noise

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
L = 2000 # samples of Q
N = 5 # Number of players
alpha = 0.5
beta = 1
std = 2
mu = 0

# Generate Q and Q_noise
Q, B, Q_noise = generate_Q_B(N, L, alpha, beta, std_Q=std, mu_Q=mu, subMean=False)

# Repeat for N players
B = np.repeat(B[:, np.newaxis, :, :], N, axis=1)
Q = np.repeat(Q[:, np.newaxis, :, :], N, axis=1)
Q_noise = np.repeat(Q_noise[:, np.newaxis, :, :], N, axis=1)

# Inverse Matrices
Q_inv = np.linalg.pinv(Q)
Q_inv_noise = np.linalg.pinv(Q_noise)

# Calculate X action of all three methods
x_opt = np.matmul(-Q_inv, B)
x_QNoise = np.matmul(-Q_inv_noise, B)
x_Noise = x_opt + std * np.random.randn(L, N, N, 1) + mu

# Calculate cost
cost_optimal = calculate_scores(Q, B, x_opt, L)
cost_Q_Noise = calculate_scores(Q, B, x_QNoise, L)
cost_X_noise = calculate_scores(Q, B, x_Noise, L)

# Sum on L games
cost_optimal = np.sum(cost_optimal)
cost_Q_Noise = np.sum(cost_Q_Noise)
cost_X_noise = np.sum(cost_X_noise)

# Plot results
t = np.ones(5)
plt.plot(t, t * cost_optimal, '--k', label='Opt')
plt.plot(t, t * cost_Q_Noise, label='Q Noise')
plt.plot(t, t * cost_X_noise, label='X Noise')
plt.legend(), plt.ylabel("# Cost"), plt.show()

print("Finsh !!! !")