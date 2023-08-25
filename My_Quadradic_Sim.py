# -*- coding: utf-8 -*-
"""
Created on : ------

@author: Ariel_Kantorovich
"""
import numpy as np
import matplotlib.pyplot as plt

# %% Setting Functions

def objective_global_cost(d, N=0):
    """
    This function calculate the global objective function of all agents where we randomize Q and B
    :param d: dim of action 'x' scaler
    :param N: Number of agents
    :return: cost - scaler
    """
    q = np.random.rand(d, d)
    Q = q @ q.T # Generate Q symmetric and self positve
    B = np.random.rand(d, 1)
    return float(-0.5 * B.T @ np.linalg.inv(Q) @ B)

def compute_cost(Qn, Bn, x):
    """
    compute the cost for each agent given their specific Qn and Bn matrices
    :param Qn: size dxd
    :param Bn: size dx1
    :param x: size dx1
    :return: cost-scalar
    """
    return 0.5 * x.T @ Qn @ x + Bn.T @ x

def compute_gradient(Qn, Bn, x):
    """
    Compute the gradient of the cost function for each agent, Pay attention our assumption is that Q is symmetric
    :param Qn: size dxd
    :param Bn: size dx1
    :param x: size dx1
    :return: derivative size dx1
    """
    return Qn @ x + Bn


def multi_agent_gradient_descent(Qn, Bn, N, d, learning_rate, T):
    """
    Multi-Agent Gradient Descent function
    :param N: Number of agents
    :param d: Size of action 'x'
    :param learning_rate:
    :param T:
    :return:
    """
    # Initialize agents' variables 'x' with random values
    x_agents = 0.1 * np.ones((N, d, 1))           #np.random.rand(N, d, 1)
    # Initialize cost agents
    cost_Matrix = np.zeros((N, T, 1))
    # Init temp Gradient
    gradient_parallel = np.zeros((N, d, 1))
    # Hyper_param for normalization
    epsilon_std = 1e-7
    # Loop over the iterations
    for t in range(T):
        # Loop over each agent
        for agent_idx in range(N):
            # Compute Quadratic cost
            cost_Matrix[agent_idx, t, :] = compute_cost(Qn[agent_idx], Bn[agent_idx], x_agents[agent_idx])
            # Compute the gradient for the current agent and update temp variable
            gradient = compute_gradient(Qn[agent_idx], Bn[agent_idx], x_agents[agent_idx])
            gradient_parallel[agent_idx] = gradient
        # Update the agent's variable 'x' using gradient descent
        x_agents -= learning_rate * gradient_parallel
        # Project the action to [-1,1] (Normalization)
        mean_values = np.mean(x_agents, axis=(1, 2), keepdims=True)
        std_values = np.std(x_agents, axis=(1, 2), keepdims=True)
        x_agents = (x_agents - mean_values) / (std_values + epsilon_std)
    final_cost = np.sum(cost_Matrix, axis=0).squeeze()
    return x_agents, final_cost

# %% Parameters and Starting progrram

# Parameters
N = 5  # Number of agents
d = 3  # dim of action 'x'
learning_rate = 0.01
T = 600 # Number of iteration
N_exper = 5 # Number of Experiments for Qn, Bn
iteration = np.arange(T)

Objective_results = objective_global_cost(d, N)

# %% Main loop

plt.plot(iteration, Objective_results * np.ones(T), '--k', label="Objective")
for n in range(N_exper):
    # Randomize Qn and Bn for each agent (for simplicity, using random matrices)
    q = np.random.rand(N, d, d)
    Qn = q + q.transpose(0, 2, 1)  # Generate Q symmetric
    Bn = np.random.rand(N, d, 1)
    resulting_x_agents, result_cost_Matrix = multi_agent_gradient_descent(Qn, Bn, N, d, learning_rate, T)
    print(f"Trial {n+1} Finish !")
    # Plot results
    plt.plot(iteration, result_cost_Matrix, label=f"trial={n+1}"), plt.xlabel("# Number of Iteration"), plt.ylabel("Cost function all agent"),
    plt.legend()

plt.show()

