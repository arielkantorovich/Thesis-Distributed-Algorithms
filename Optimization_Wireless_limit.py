import numpy as np
import scipy.optimize as optim

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


# Define Optimization Parameters
N = 5
N0 = 0.001
alpha = 10e-3
L = 1
add_gain = True
add_gain_param = 10.0
g = generate_gain_channel(L, N, alpha)
# Add Gain to transmiter channel
if add_gain:
    g_channel = add_gain_param * np.eye(N)
    g = g + g_channel
g_square = g ** 2
P = np.random.rand(L, N, 1)  # Generate Power from uniform distributed
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

def objective_function(P, *args):
    """
    We define the sum with minus because in general scipy optimization algorithms try to minimize objective
    :param P:
    :param g:
    :param I:
    :param N0:
    :return:
    """
    global g_square, In, N0  # Access global variables g, In, and N0
    g, I, N00 = args
    return -np.sum(np.log2(1 + (g * P) / (I + N00)))


# Define bounds for P, where each P_n should be between 0 and 1 (Optimization constraint)
bounds = [(0, 1)] * N  # Create a list of N tuples, each with bounds (0, 1)
result = optim.differential_evolution(objective_function, bounds, args=(g_square, In, N0))

# Extract the optimal power allocation from the result
optimal_power_allocation = result.x
optimal_objective_value = -result.fun
# Print the results
print("Optimal power allocation:", optimal_power_allocation)
print("Optimal value of the objective function:", optimal_objective_value)

print("Check")