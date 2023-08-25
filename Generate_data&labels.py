# -*- coding: utf-8 -*-
"""
Created on : ------

@author: Ariel_Kantorovich
"""
import numpy as np


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
    X_train = np.concatenate((points_temp, mean_dist), axis=1)
    Q_new = np.repeat(Q, N, axis=0)
    Y_train = Q_new.reshape(L*N, N**2, 1)
    # X_train = Y_train # Option learn Identity matrix
    return Q, B, X_train, Y_train



# Initialize Parameters:
alpha = 4.0# Hyperparam for exp matrix Q
beta = 1 # Hyperparam for matrix B
N = 5 # Number of players
L = 50 # trials for Q and B
Q, B, X_train, Y_train = generate_Q_B(N, L, alpha, beta, std_Q=0, mu_Q=0, subMean=True)

# Save arrays
np.save("Numpy_array_save/Q.npy", Q)
np.save("Numpy_array_save/x_train.npy", X_train)
np.save("Numpy_array_save/y_train.npy", Y_train)

print("Finsh!!")