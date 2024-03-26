# -*- coding: utf-8 -*-
"""
Created on : ------

@author: Ariel_Kantorovich
"""
import matplotlib.pyplot as plt
import numpy as np
from numba import jit

@jit(nopython=True)
def calculate_scores(Q, B, x_L, L):
    """
    :param Q: size (L,NxN)
    :param B: size (L, Nx1)
    :param x_L: size (L, Nx1)
    :param L: scalar
    :return: scores of L games
    """
    scores = 0
    for i in range(L):
        x = x_L[i]
        scores += np.sum(0.5 * np.transpose(x) @ Q[i] @ x + np.transpose(x) @ B[i])
    return scores


def generate_DataSet(N, L, alpha, beta, Test_Flag=False):
    """
    :param N: Number of players
    :param L: Trials to experiment
    :param alpha: scalar
    :param beta:  scalar
    :param Test_Flag: bool, if True return Q and B
    :return: Q, B where Q is prob distance map size (L, N, N) and B is one vector size (L, N, 1)
    """
    # Step 1: Generate L different sets of N random points
    points = np.random.rand(L, N, 2)
    # Step 2: Build distance maps D for all trials
    points_expanded = points[:, np.newaxis, :, :]
    differences = points_expanded - points[:, :, np.newaxis, :]
    distances = np.linalg.norm(differences, axis=-1)
    # Step 3: Generate Q = exp(-alpha*D) for all trials
    Q = np.exp(-alpha * distances)
    # Generate B of size Nx1xL
    B = np.random.uniform(low=-beta, high=beta, size=(L, N, 1))
    # Build X_train data set
    points_temp = points.reshape(L*N, 2)
    mean_dist = np.mean(distances, axis=2)
    mean_dist = mean_dist.reshape(L*N, 1)
    # Generate M neighbors that close
    M = 2
    sorted_rows = np.sort(distances, axis=-1)
    second_smallest_distances = sorted_rows[:, :, 1:(M+1)]
    small_dist = second_smallest_distances.reshape(N * L, M)
    # Finally build X train
    X_train = np.concatenate((points_temp, mean_dist, small_dist), axis=1)
    X_train = X_train.reshape(L * N, X_train.shape[1], 1)
    # Build Y_train
    Y_train_Nash = Q.reshape(L*N, N, 1)
    # Build Xopt
    x_opt = -np.matmul(np.linalg.pinv(Q), B)
    Y_train_x = x_opt.reshape(L*N, 1, 1)
    if Test_Flag:
        return X_train, Y_train_Nash, Y_train_x, Q, B
    return X_train, Y_train_Nash, Y_train_x

def generate_linear_estimator(X_train, Y_train_nash, Y_train_x):
    """
    The Function calculate the Linear Estimator LMMSE matrices
    :param X_train: (N*L, A, 1)
    :param Y_train_nash: (N*L, N, 1)
    :param Y_train_x: (N*L, 1, 1)
    :return: W_nash: size (N, A), W_xopt: size(1, A)
    """
    # Transpose x
    XT = np.transpose(X_train, axes=(0, 2, 1))
    # Calculate COV matrices
    cov_yx = np.sum(np.matmul(Y_train_nash, XT), axis=0)
    cov_xx = np.sum(np.matmul(X_train, XT), axis=0)
    # Calculate W nash
    cov_inv_xx = np.linalg.pinv(cov_xx)
    W_nash = np.matmul(cov_yx, cov_inv_xx)
    # Calculate COV for X
    cov_yx_opt = np.sum(np.matmul(Y_train_x, XT), axis=0)
    W_x = np.matmul(cov_yx_opt, cov_inv_xx)
    return W_nash, W_x




########################## Main Code: ###############################
# %% Parameters
L_test = 600
L_train = [10, 1000, 10000, 100000, 500000, 1000000]
N = 5 # Number of players
alpha = 4.0
beta = 0.5
Border_projection = 2
Error_test_nash = []
Error_test_xopt = []
global_error_nash = []
global_error_xopt = []
X_test, Y_test_Nash, Y_test_x, Q, B = generate_DataSet(N, L_test, alpha, beta, True)

# Calculate optimal cost
BT = np.transpose(B, axes=(0, 2, 1))
x_opt = - np.matmul(np.linalg.pinv(Q), B)
# Project solution
x_opt = np.minimum(np.maximum(x_opt, -Border_projection), Border_projection)
C_opt = calculate_scores(Q, B, x_opt, L_test)

for L in L_train:
    # Generate Data Set
    X_train, Y_train_Nash, Y_train_x = generate_DataSet(N, L, alpha, beta)
    # Calculate Linear - Estimators
    W_nash, W_x = generate_linear_estimator(X_train, Y_train_Nash, Y_train_x)
    # Repeat W for element-wise calculation
    W_nash_repeat = np.repeat(W_nash[np.newaxis, :, :], N*L_test, axis=0)
    W_x_repeat = np.repeat(W_x[np.newaxis, :, :], N*L_test, axis=0)
    # Calculate output from Estimator
    Y_nash = np.matmul(W_nash_repeat, X_test)
    Y_x = np.matmul(W_x_repeat, X_test)
    Error_nash = np.mean(np.linalg.norm(Y_nash - Y_test_Nash, axis=1))
    Error_xopt = np.mean(np.linalg.norm(Y_x - Y_test_x, axis=1))
    Error_test_nash.append(Error_nash)
    Error_test_xopt.append(Error_xopt)
    # Calculate vectors
    x_opt_y = Y_x.reshape(L_test, N, 1)
    Q_nash = Y_nash.reshape(L_test, N, N)
    x_nash = - np.matmul(np.linalg.pinv(Q_nash), B)
    # Project vectors to their constraint
    x_opt_y = np.minimum(np.maximum(x_opt_y, -Border_projection), Border_projection)
    x_nash = np.minimum(np.maximum(x_nash, -Border_projection), Border_projection)
    # Calculate error
    nash_error = calculate_scores(Q, B, x_nash, L_test)
    xopt_error = calculate_scores(Q, B, x_opt_y, L_test)
    # Finally save error results
    global_error_nash.append(nash_error)
    global_error_xopt.append(xopt_error)


# Plot results
plt.figure(1)
plt.plot(L_train, Error_test_nash, '-*', label='Error Nash'),
plt.plot(L_train, Error_test_xopt, '-*', label='Error Xopt'),
plt.legend(), plt.title("Test Error LMMSE"), plt.xlabel("# Training Set"), plt.ylabel("# Error L2")
plt.figure(2)
plt.plot(L_train, global_error_nash, '-*', label='C(Nash)'),
plt.plot(L_train, global_error_xopt, '-*', label='C(X)'),
plt.plot(L_train, C_opt * np.ones(len(L_train)), '--k', label='C(opt)'),
plt.legend(), plt.title("Error from Optimal"), plt.xlabel("# Training Set"), plt.ylabel("# Error MSE"),
plt.show()

print("Finsh !!!!!")



