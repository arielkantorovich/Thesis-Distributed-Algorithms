import numpy as np
import scipy.optimize as optim
from numba import jit
import matplotlib.pyplot as plt

np.random.seed(0) # Set Initialize Vector

def generate_gain_channel(L, N, alpha):
    """
    :param L: (int) trials of game
    :param N:  (int) number of players
    :param alpha:  (int) some constant
    :return: g return the gain of channels between players
    """
    # Generate random transceiver locations in a radius of 1
    Transreceivers = np.random.rand(L, N, 2) * 2 - 1  # Scale to (-1, 1) and then to (-1, 1) radius 1
    # Generate random receiver locations in a radius of 0.1 around each transceiver
    Rlink = 0.1
    ReceiverOffsets = Rlink * (np.random.rand(L, N, 2) * 2 - 1)  # Scale to (-1, 1) and then to (-0.1, 0.1)
    Receivers = Transreceivers + ReceiverOffsets
    # Calculate distances between transceivers and receivers
    distances = np.linalg.norm(Transreceivers[:, :, np.newaxis, :] - Receivers[:, np.newaxis, :, :], axis=3)
    g = alpha / (distances ** 2)
    return g

@jit(nopython=True)
def calc_second_gradient(N, gradients_second, g_diag, g_square, P, In):
    """
    :param N: int number of Players
    :param gradients_second: zero numpy array size (L, N, 1)
    :param g_diag: numpy array size (L, N, 1)
    :param g_square: numpy array size (L, N, N)
    :param P: numpy array size (L, N, 1)
    :param In: numpy array size (L, N, 1)
    :return: gradients_second: numpy array size (L, N, 1)
    """
    N0 = 0.001
    # L, _, _ = gradients_second.shape
    for n in range(N):
        for j in range(N):
            if n != j:
                gradients_second[n, 0] += (
                        g_diag[j, 0] * g_square[n, j] * P[j, 0] /
                        ((In[j, 0] + N0) * (In[j, 0] + N0 + g_diag[j, 0] * P[j, 0]))
                )
    return -1.0 * gradients_second
def multi_wireless_loop(N, L, T, g, lr, beta, P):
    """
    :param N: (int) number of players
    :param L: (int) trials of game
    :param T: (int) duration of experiment
    :param g: (L, N, N) channel gain matrix
    :param lr: (T, ) learning rate vector
    :return: mean_P_record (T, N) the P array with record on time
    """
    # Initialize Parameters
    Border_floor = 0
    Border_ceil = 1
    N0 = 0.001
    g_square = g ** 2
    # Initialize record variables
    P_record = np.zeros((T, N, 1))
    global_objective = np.zeros((T, ))
    gradients_record = np.zeros((T, N, 1))
    # Prepare g to calculation
    g_diag = np.diagonal(g_square, axis1=1, axis2=2)
    g_diag = g_diag.reshape(N, 1)
    g_colum = np.transpose(g_square, axes=(0, 2, 1))
    # Initialize gradients array
    gradients_first = np.zeros((N, 1))
    gradients_second = np.zeros((N, 1))
    # squeezze
    g_square = np.squeeze(g_square, axis=0)
    g_colum = np.squeeze(g_colum, axis=0)
    for t in range(T):
        # calculate instance
        In = np.matmul(g_colum, P) - g_diag * P
        # calculate gradients
        numerator = (g_diag / (In + N0))
        gradients_first = (numerator / (1 + numerator * P)) - beta
        ###########################################################################################################################################
        # calcaulate the second gradient
        gradients_second = calc_second_gradient(N, gradients_second, g_diag, g_square, P, In)
        # update agent vector(P)
        P += lr[t] * (gradients_first + gradients_second)
        # Project the action to [Border_floor, Border_ceil] (Normalization)
        P = np.minimum(np.maximum(P, Border_floor), Border_ceil)
        # Save results in record
        P_record[t] = P
        gradients_record[t] = gradients_first + gradients_second
        # Calculate global objective
        temp = np.log(1 + numerator * P) - beta * P
        temp = temp.squeeze()
        global_objective[t] = np.sum(temp, axis=0)
    # Finally Let's mean for all L trials
    P_record = P_record.squeeze()
    gradients_record = gradients_record.squeeze()
    return P_record, global_objective, gradients_record

def objective_function(P, *args):
    """
    We define the sum with minus because in general scipy optimization algorithms try to minimize objective
    :param P:
    :param g:
    :param I:
    :param N0:
    :return:
    """
    g_diag, g_colum, N0 = args
    P = P.reshape(P.shape[0], 1)
    I = np.matmul(g_colum, P) - g_diag * P
    numerator = ((g_diag * P) / (I + N0))
    x = np.sum(np.log(1 + numerator))
    return -1 * x


# Define Optimization Parameters
beta = 0.0
N = 5
N0 = 0.001
alpha = 10e-3
L = 1
T = 80000
# learning_rate = 0.03 * np.reciprocal(np.power(range(1, T + 1), 0.65))
learning_rate = 0.0001 * np.ones((T, ))
add_gain = True
add_gain_param = 10.0
g = generate_gain_channel(L, N, alpha)
# Add Gain to transmiter channel
if add_gain:
    g_channel = add_gain_param * np.eye(N)
    g = g + g_channel
g_square = g ** 2
P = 0.01 * np.random.rand(L, N, 1)  # Generate Power from uniform distributed
# Prepare g to calculation
g_diag = np.diagonal(g_square, axis1=1, axis2=2)
g_diag = g_diag.reshape(L, N, 1)
g_colum = np.transpose(g_square, axes=(0, 2, 1))
# calculate instance
In = np.matmul(g_colum, P) - g_diag * P
# Prepare Vectors to optimization Function
In = np.squeeze(In, axis=0)
P = np.squeeze(P, axis=0)
g_square = np.squeeze(g_square, axis=0)
g_diag = np.squeeze(g_diag, axis=0)


# Define bounds for P, where each P_n should be between 0 and 1 (Optimization constraint)
bounds = [(0, 1)] * N  # Create a list of N tuples, each with bounds (0, 1)
# result = optim.minimize(objective_function, P, args=(g_diag, In, N0), bounds=[(0, 1)] * N)
result = optim.differential_evolution(objective_function, bounds, args=(g_diag, np.squeeze(g_colum, axis=0), N0), maxiter=10000)

# Extract the optimal power allocation from the result
optimal_power_allocation = result.x
optimal_objective_value = -result.fun
# Print the results
print("Optimal power allocation:", optimal_power_allocation)
print("Optimal value of the objective function:", optimal_objective_value)


# run my loop code
P_record, global_record, grad_record = multi_wireless_loop(N, L, T, g, learning_rate, beta, P)


# Plot results
t = np.arange(T)
plt.figure(1)
for n in range(N):
    Pn = P_record[:, n]
    plt.plot(t, Pn, label=f"P{n}"), plt.xlabel("# Iteration"),
    plt.ylabel("# candidate action"), plt.legend()

plt.figure(2)
plt.plot(t, global_record), plt.xlabel("# Iteration"), plt.ylabel("Global Objective")

plt.figure(3)
for n in range(N):
    grad_n = grad_record[:, n]
    plt.plot(t, grad_n, label=f"grad{n}"), plt.xlabel("# Iteration"), plt.ylabel("gradients")
plt.show()
print("Finsh !!!")