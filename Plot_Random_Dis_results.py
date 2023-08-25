import numpy as np
import matplotlib.pyplot as plt

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

# Load Parameters from Experiment:
Q = np.load("Numpy_array_save/Q.npy")
x_record = np.load("Numpy_array_save/x_record.npy")
cost_record = np.load("Numpy_array_save/cost_record.npy")

# Extract size and parameters
T, N_exper, N, _ = x_record.shape
L = Q.shape[0]
beta = 1
B = beta * np.ones((L, N, 1))

# Print Statistic
candadite_score = calculate_scores(Q, B, x_record[-1, :], N_exper, N, L)
n = np.arange(N_exper)
i_optimal = np.argmin(candadite_score[1:]) + 1
print(f" Min_value = {np.min(candadite_score[1:])} \n Max_value = {np.max(candadite_score[1:])} \n Avg_value = {np.mean(candadite_score[1:])}")


# Plot graph
t = np.arange(T)
plt.plot(t, cost_record[:, 0], '--k', label='Optimum limit')
plt.plot(t, cost_record[:, i_optimal], label=f"Guess* = {i_optimal}")
plt.plot(t, cost_record[:, 50], label="Guess=50")
plt.plot(t, cost_record[:, 2], label="Guess=2")
plt.plot(t, cost_record[:, 90], label="Guess=90")
plt.xlabel("# Iteration"), plt.ylabel("candadite_score"), plt.legend()
plt.show()
