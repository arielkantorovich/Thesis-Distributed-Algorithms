import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define some seed
np.random.seed(0)

class enerrgy_optimization__package:
    def __init__(self, L, N, K, gamma_n, gamma_nk):
        self.L = L
        self.N = N
        self.K = K
        self.gamma_n = gamma_n
        self.gamma_nk = gamma_nk

    def constraint_gamma_n(self, x):
        x = x.reshape(self.N, self.K)
        constraint_value = self.gamma_n - np.sum(x, axis=1)
        return constraint_value

    # Define the total reward function (sum of all players' rewards)
    def total_reward(self, x, a_k, b_k):
        x = x.reshape(self.N, self.K)  # Reshape x to be (N, K)
        S_k = np.sum(x, axis=0)  # Sum of actions across players for each resource
        total_reward = 0
        for n in range(self.N):
            log_term = np.sum(np.log(1 + x[n]))
            quadratic_term = np.sum(a_k[n] * x[n] * S_k ** 2)
            linear_term = np.sum(b_k[n] * x[n] * S_k)
            total_reward += log_term - (quadratic_term + linear_term)
        return -total_reward  # We negate because we want to maximize

    def main_package_loop(self, x_init, a_k, b_k):
        # Solve the optimization problem for L games
        results = np.zeros((self.L, ))
        optimized_actions = np.zeros((self.L, self.N, self.K))
        for game in range(self.L):
            # Initial guess
            x0 = x_init[game]
            # Bounds: x_nk must be between 0 and 1
            bounds = [(0, self.gamma_nk) for _ in range(self.K * self.N)]
            # Constraints: sum of each player's actions must be less than or equal to gamma_n
            constraints = [{'type': 'ineq', 'fun': lambda x: self.constraint_gamma_n(x)}]
            # Minimize the negative total reward function
            result = minimize(self.total_reward, x0.flatten(), args=(a_k[game], b_k[game]), bounds=bounds,
                              constraints=constraints)
            # Store the result of this game
            results[game] = -result.fun  # Store the maximized reward (negated)
            # Reshape the optimized x_nk back to (N, K) and store it
            optimized_actions[game] = result.x.reshape(self.N, self.K)
        # Calculate the mean reward over all games
        mean_reward = np.mean(results)
        return mean_reward, optimized_actions

def project_onto_simplex(V, z=1):
    """
    Project matrix X onto the probability simplex.
    :param V: numpy array, matrix of size (LxNxK)
    :param z: integer, norm of projected vector
    :return X: numpy array, matrix of the same size as X, projected onto the probability simplex
    """
    L, N, K = V.shape
    U = np.sort(V, axis=-1)[:, :, ::-1]
    z_vector = np.ones((L, N)) * z
    cssv = np.cumsum(U, axis=-1) - z_vector[:, :, np.newaxis]
    ind = np.arange(K) + 1
    cond = U - cssv / ind > 0
    rho = np.count_nonzero(cond, axis=-1)
    theta = cssv[np.arange(L)[:, np.newaxis], np.arange(N), rho - 1] / rho
    result = np.maximum(V - theta[:, :, np.newaxis], 0)
    return result

def calculate_NE_gradient(Xn_k, A_k, B_k, Sk):
    """
    The function return vectorization gradient of the player's grad_rnk
    :param Xn_k: np array size (L, N, K)
    :param A_k: np array size (L, N, K)
    :param B_k: np array size (L, N, K)
    :param Sk: np array size (L, N, K)
    :return: grad_r_nk: np array size (L, N)
    """
    grad_Pk = A_k * (Sk ** 2 + 2 * Sk * Xn_k) + B_k * (Sk + Xn_k)
    grad_vk = 1 / (1 + Xn_k)
    # grad_r_nk = np.sum(grad_vk - grad_Pk, axis=2)
    grad_r_nk = grad_vk - grad_Pk
    return grad_r_nk

def calculate_residual_gradient(Xn_k, A_k, B_k, Sk):
    """
    This function calculate the residual gradient for global reward
    :param Xn_k: (L, N, K) np.array
    :param A_k: (L, N, K) np.array
    :param B_k: (L, N, K) np.array
    :param Sk: (L, N, K) np.array
    :return: residual gradient (L, N, K) np.array
    """
    temp = - (2 * A_k * Xn_k * Sk + B_k * Xn_k)
    grad_sum = np.sum(temp, axis=1)
    grad_sum = np.repeat(grad_sum[:, np.newaxis, :], N, axis=1)
    return grad_sum - temp

def main_loop(N, T, gamma_n_k, gamma_n, save_grad_debug, learning_rate,
              Xn_k, A_k, B_k,
              reward_list, grad_list, is_global=False):
    """
    :param N: (int) number of players
    :param T: (int) number of iteration
    :param gamma_n_k: (int)
    :param gamma_n: (int)
    :param save_grad_debug: (bool)
    :param learning_rate: (np.array) size (T, )
    :param Xn_k: (np.array) size (L, N, K)
    :param A_k: (np.array) size (L, N, K)
    :param B_k: (np.array) size (L, N, K)
    :param is_global: (bool)
    :return:
    """
    grad_Xnk_global = np.zeros((L, N, K))
    for t in range(T):
        # Calculate sum of action and duplicate for vectorization operation
        Sk = np.sum(Xn_k, axis=1)
        Sk = np.repeat(Sk[:, np.newaxis, :], N, axis=1)
        # Calculate reward
        V = np.log(1 + Xn_k)
        P = A_k * Xn_k * (Sk ** 2) + B_k * Xn_k * Sk
        r_n = np.sum(V - P, axis=2)
        reward_list[t] = np.mean(r_n, axis=0)
        # Gradient ascent
        grad_Xnk_NE = calculate_NE_gradient(Xn_k, A_k, B_k, Sk)
        if is_global:
            grad_Xnk_global = calculate_residual_gradient(Xn_k, A_k, B_k, Sk)
        total_grad = grad_Xnk_NE + grad_Xnk_global
        Xn_k = Xn_k + learning_rate[t] * total_grad
        if save_grad_debug:
            temp = np.sum(total_grad, axis=2)
            grad_list[t] = np.mean(temp, axis=0)
        # Project to action to the set:
        Xn_k = np.clip(Xn_k, a_min=0, a_max=gamma_n_k)
        Xn_k = project_onto_simplex(Xn_k, z=gamma_n)


    return Xn_k, reward_list, grad_list


# Initialize constant parameters
L = 10
K = 24
N = 5
T = 100
gamma_n_k = 0.1
gamma_n = 1.4
learning_rate = 0.06 * np.reciprocal(np.power(range(1, T + 1), 0.9))
learning_rate_global = 0.06 * np.reciprocal(np.power(range(1, T + 1), 0.9))

# Initialize Xnk
Xn_k = np.random.uniform(low=0.0, high=1.0, size=(L, N, K))
Xn_k = np.clip(Xn_k, a_min=0, a_max=gamma_n_k)
Xn_k = project_onto_simplex(Xn_k, z=gamma_n)


# define X nash and x global
X_NE = Xn_k.copy()
X_global = Xn_k.copy()
x_global_package = Xn_k.copy()

# Game initialize Parameters
alpha = 1.5
beta = 0.97
A_k = alpha * np.random.uniform(low=0.1, high=1.8, size=(L, N, K))
B_k = beta * np.random.uniform(low=0.0, high=5.0, size=(L, N, K))

# Save results
save_grad_debug = True
reward_list_NE = np.zeros((T, N))
grad_list_NE = np.zeros((T, N))
reward_list_global = np.zeros((T, N))
grad_list_global = np.zeros((T, N))

# Read to main loop
X_NE, reward_list_NE, grad_list_NE = main_loop(N, T, gamma_n_k,
                                               gamma_n, save_grad_debug, learning_rate,
                                               X_NE, A_k, B_k,
                                               reward_list_NE, grad_list_NE, is_global=False)

X_global, reward_list_global, grad_list_global = main_loop(N, T, gamma_n_k,
                                                           gamma_n, save_grad_debug, learning_rate_global,
                                                           X_global, A_k, B_k,
                                                           reward_list_global, grad_list_global, is_global=True)
# Get Solution from optimization package
optim_class = enerrgy_optimization__package(L, N, K, gamma_n, gamma_n_k)
reward_mean_package, x_global_package = optim_class.main_package_loop(x_global_package, A_k, B_k)

# Plot Section
t = np.arange(T)
for n in range(N):
    r_n_NE = reward_list_NE[:, n]
    grad_n_NE = grad_list_NE[:, n]
    r_n_global = reward_list_global[:, n]
    grad_n_global = grad_list_global[:, n]
    plt.figure(1)
    plt.plot(t, r_n_NE, label=f"r_{n} NE"), plt.xlabel("# Iteration"), plt.legend()
    plt.figure(2)
    plt.plot(t, grad_n_NE, label=f"grad_{n} NE"), plt.xlabel("# Iteration"), plt.legend()
    plt.figure(3)
    plt.plot(t, r_n_global, label=f"r_{n} global"), plt.xlabel("# Iteration"), plt.legend()
    plt.figure(4)
    plt.plot(t, grad_n_global, label=f"grad_{n} global"), plt.xlabel("# Iteration"), plt.legend()


total_NE_reward = np.sum(reward_list_NE, axis=1)
total_global_reward = np.sum(reward_list_global, axis=1)
plt.figure(5)
plt.plot(t, reward_mean_package * np.ones(T), '--k', label="$\sum_{n} r_{n} Optim Package$"), plt.xlabel("# Iteration"), plt.legend()
plt.plot(t, total_NE_reward, label="$\sum_{n} r_{n} NE$"), plt.xlabel("# Iteration"), plt.legend()
plt.plot(t, total_global_reward, label="$\sum_{n} r_{n} global$"), plt.xlabel("# Iteration"), plt.legend()
plt.show()

print("Finsh ..... ")


