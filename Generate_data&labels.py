# -*- coding: utf-8 -*-
"""
Created on : ------

@author: Ariel_Kantorovich
"""
import numpy as np

def generate_Q_d_oneRank(N, L, mu=0, std=1, d=3, SubMean=True):
    """
    :param d: number of vector one rank (int)
    :param N: number of players in game (int)
    :param d: number of vector one rank (int)
    :param L: number of trials (int)
    :param mu: mean for gaussian noise (int)
    :param std: standard division for gaussian noise (int)
    :return: X_train size (NL, N, 1)
             Y_train size (NL, N^2, 1)
             Q_save size (L, N, N)
    """
    # Generate Q_save size (L, N, N)
    u = np.random.rand(L, d, N, 1)
    u_transposed = u.transpose((0, 1, 3, 2))
    Q_save = np.matmul(u, u_transposed)
    Q_save = np.sum(Q_save, axis=1) # sum on d
    # Expand u to N players congestion game take care onle d vector we see
    repeated_u = np.repeat(u, N // d, axis=1)
    u = repeated_u.reshape(L * N, N, 1)
    u_transposed = u.transpose((0, 2, 1))
    Q = np.matmul(u, u_transposed)
    if SubMean:
        # In this part we subtract the mean from every L trial in Q
        trial_means = np.mean(Q, axis=(1, 2))
        # Expand trial_means to have the same shape as Q (L, N, N)
        trial_means_expanded = trial_means[:, np.newaxis, np.newaxis]
        Q = Q - trial_means_expanded
    noise = std * np.random.randn(L*N, N, 1) + mu
    u_hat = u + noise
    # X_train = u_hat
    X_train = np.matmul(u_hat, u_hat.transpose((0, 2, 1)))
    Y_train = Q.reshape(L*N, N**2, 1)
    return X_train, Y_train, Q_save

def generate_Q_oneRank(N, L, mu=0, std=1):
    """
    :param N: number of players in game (int)
    :param L: number of trials (int)
    :param mu: mean for gaussian noise (int)
    :param std: standard division for gaussian noise (int)
    :return: X_train size (NL, N, 1)
             Y_train size (NL, N^2, 1)
             Q_save size (L, N, N)
    """
    # Generate Q_save size (L, N, N)
    u = np.random.rand(L, N, 1)
    u_transposed = u.transpose((0, 2, 1))
    Q_save = np.matmul(u, u_transposed)
    # Expand u to N players congestion game
    u = np.repeat(u, N, axis=0)
    u_transposed = u.transpose((0, 2, 1))
    Q = np.matmul(u, u_transposed)
    noise = std * np.random.randn(L*N, N, 1) + mu
    u_hat = u + noise
    X_train = u_hat
    Y_train = Q.reshape(L*N, N**2, 1)
    B = np.ones((L*N, N, 1))
    Y_train_2 = -np.matmul(np.linalg.pinv(Q), B)
    # Y_train_2 = Q
    return X_train, Y_train, Q_save, Y_train_2


def generate_Q_B(N, L, alpha, beta, std_Q=0, mu_Q=0, subMean=True):
    """
    :param SubMean: boolean variable if subtract mean from Q or not
    :param std_Q: variance for Q scalar
    :param mu_Q: mu for Q scalar
    :param N: Number of players
    :param L: Trials to experiment
    :param alpha: scalar
    :param beta:  scalar
    :return: Q, B where Q is prob distance map size (L, N, N) and B is one vector size (L, N, 1)
    """
    # Step 1: Generate L different sets of N random points
    points = np.random.rand(L, N, 2)
    # Step 2: Build distance maps D for all trials
    points_expanded = points[:, np.newaxis, :, :]
    differences = points_expanded - points[:, :, np.newaxis, :]
    distances = np.linalg.norm(differences, axis=-1)
    # Step 3: Generate Q = exp(-alpha*D) for all trials
    Q = np.exp(-alpha * distances) + std_Q * np.random.randn(L, N, N) + mu_Q
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
    # Build X_train data set
    # X_train = Q.reshape(L * N, N, 1)  # Option of distances of Q
    # X_train = distances.reshape(L * N, N, 1) # Option of distances
    points_temp = points.reshape(L*N, 2)
    mean_dist = np.mean(distances, axis=2)
    mean_dist = mean_dist.reshape(L*N, 1)
    # Generate M neighbors that close
    M = 3
    sorted_rows = np.sort(distances, axis=-1)
    second_smallest_distances = sorted_rows[:, :, 0:(M+1)]
    small_dist = second_smallest_distances.reshape(N * L, M+1)
    # Finally build X train
    X_train = np.concatenate((mean_dist, small_dist), axis=1)
    # X_train = np.concatenate((points_temp, mean_dist, small_dist), axis=1)
    Q_new = np.repeat(Q, N, axis=0)
    Y_train = Q_new.reshape(L*N, N**2, 1)
    # X_train = Y_train # Option learn Identity matrix
    B_expand = np.ones((N*L, N, 1))
    Y_train_2 = -np.matmul(np.linalg.pinv(Q_new), B_expand)
    return Q, B, X_train, Y_train, Y_train_2



# Initialize Parameters:
alpha = 4.0# Hyperparam for exp matrix Q
beta = 1 # Hyperparam for matrix B
N = 7 # Number of players
L = 100 # trials for Q and B
d = 5
mu = 0
std = 1

# X_train, Y_train, Q, Y_train_2 = generate_Q_oneRank(N, L, mu, std)
Q, B, X_train, Y_train, Y_train_2 = generate_Q_B(N, L, alpha, beta, std_Q=0, mu_Q=0, subMean=True)
#
# Save arrays
np.save("Numpy_array_save/Q.npy", Q)
np.save("Numpy_array_save/x_test.npy", X_train)
np.save("Numpy_array_save/y_test_Nash.npy", Y_train)
np.save("Numpy_array_save/y_test_No_Nash.npy", Y_train_2)

print("Finsh!!")