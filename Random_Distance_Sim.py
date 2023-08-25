# -*- coding: utf-8 -*-
"""
Created on : ------

@author: Ariel_Kantorovich
"""
import numpy as np
import matplotlib.pyplot as plt

# %% Setting Functions
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
    return Q, B

def generate_Qn_Bn(N_exper, N, d):
    """
    :param N_exper:
    :param N:
    :param d:
    :return: Qn, Bn
    """
    Qn = np.random.rand(N_exper, N, N) # np.random.rand(N_exper, N, d)
    Bn = np.ones((N_exper, N, 1))  # np.ones((N_exper, N, d))
    return Qn, Bn

def compute_gradient(Qn, Bn, x):
    """
    Compute the gradient of the cost function for each agent, Pay attention our assumption is that Q is symmetric
    :param Qn: size dxd
    :param Bn: size dx1
    :param x: size dx1
    :return: derivative size dx1
    """
    return np.matmul(Qn, x) + Bn


def multi_agent_gradient_descent(N_exper, N, T, learning_rate,
                                 Qn, Bn, recordFlag, Q, B, L):
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
    # Initialize x_agents
    x_agents = 0.1 * np.ones((N_exper, N, 1))
    # Initialize x_agents and cost to track
    x_record = np.zeros((T, N_exper, N, 1))
    cost_record = np.zeros((T, N_exper, ))
    # loop over time
    for t in range(T):
        # Compute gradients in parallel
        gradients = compute_gradient(Qn, Bn, x_agents)
        # Update the agent's variable 'x' using gradient descent
        x_agents -= learning_rate[t] * gradients
        # Project the action to [-1,1] (Normalization)
        x_agents = np.minimum(np.maximum(x_agents, -1), 1)
        if recordFlag:
            x_record[t] = x_agents
            cost_record[t] = calculate_scores(Q, B, x_agents, N_exper, N, L)
    return x_agents, x_record, cost_record


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
    scores = np.zeros((N_exp, ))
    for i in range(N_exp):
        result_pass = 0.5 * x_optimal[i].T @ Q @ x_optimal[i] + x_optimal[i].T @ B
        scores[i] = np.mean(result_pass)
    # for i in range(N_exp):
    #     x = x_optimal[i].reshape(N, 1)
    #     score = 0
    #     for j in range(L):
    #         Q_ij = Q[j]
    #         B_ij = B[j]
    #         score += 0.5 * x.T @ Q_ij @ x + B_ij.T @ x
    #     # Calcalute the mean score of x_opt to all L trials of Q and B
    #     scores[i] = score / L
    return scores


def calculate_optimum_score(Q, B):
    """
    :param Q: (L, N, N)
    :param B: size (L, N, 1)
    :return: optimum cost  scalar
    """
    Q_inv = np.linalg.inv(Q)
    cost_per_i = -0.5 * np.transpose(B, (0, 2, 1)) @ Q_inv @ B
    return np.mean(cost_per_i)



# %% Parameters and Starting progrram
N = 5 # Number of players
d = 1 # dim of action
L = 100 # trials for Q and B
N_exper = 100 * L # trials for Qn and Bn
T = 3000 # Number of iteration
learning_rate = 0.03 * np.reciprocal(np.power(range(1, T + 1), 0.7))
alpha = 4.0# Hyperparam for exp matrix Q
beta = 1 # Hyperparam for matrix B
std_Q = 0 # Variance process noise for Q
mu_Q = 0 # Mu process noise for Q
subMean = True # Subtract mean from Q or not
recordFlag = True # track the progress of cost and agent
SaveFlag = True # Save array results to analyze

# %% Main loop
Q, B = generate_Q_B(N, L, alpha, beta, std_Q, mu_Q, subMean)
Qn, Bn = generate_Qn_Bn(N_exper, N, d)
# Generate optimum solution to guess n=0
Qn[0] = np.mean(Q, axis=0)
x_optimal, x_record, cost_record = multi_agent_gradient_descent(N_exper, N, T, learning_rate, Qn, Bn, recordFlag, Q, B, L)
candadite_score = calculate_scores(Q, B, x_optimal, N_exper, N, L)

# %% Save results + plot results
if SaveFlag:
    np.save("Numpy_array_save/x_record.npy", x_record)
    np.save("Numpy_array_save/cost_record.npy", cost_record)
    np.save("Numpy_array_save/Q.npy", Q)

n = np.arange(N_exper)
# plt.bar(n, candadite_score),\
# plt.xlabel("# N trials of Qn Bn"), plt.ylabel("# Mean L trials of Q and B"), plt.show()
# Extract Information
i_optimal = np.argmin(candadite_score[1:]) # becuase in 0 we have optimum solution
print(f" Min_value = {np.min(candadite_score)} \n Max_value = {np.max(candadite_score)} \n Avg_value = {np.mean(candadite_score)}")
print(f"Qn_optimal = {Qn[i_optimal]}")

# %% plot results2
if recordFlag:
    t = np.arange(T)
    plt.plot(t, cost_record[:, 0], '--k', label='Optimum limit')
    plt.plot(t, cost_record[:, i_optimal], label=f"guess* = {i_optimal}")
    plt.plot(t, cost_record[:, 50], label="guess=50")
    plt.plot(t, cost_record[:, 2], label="guess=2")
    plt.plot(t, cost_record[:, 90], label="guess=90")
    plt.xlabel("# Iteration"), plt.ylabel("candadite_score"), plt.legend()
    plt.show()
