import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.nn.init as init
from numba import jit
import scipy.optimize as optim

# np.random.seed(0)

# %% Setting Archticture network
class Wireless(nn.Module):
    def __init__(self, input_size, output_size):
        super(Wireless, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, output_size)  # Output layer with 1 neuron for beta prediction
        self.init_weights()

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = torch.relu(self.fc4(x))
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # init.normal_(m.weight, mean=0, std=1.0)
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


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
    L, _, _ = gradients_second.shape
    for n in range(N):
        for j in range(N):
            if n != j:
                for l in range(L):
                    gradients_second[l, n, 0] += (
                            g_diag[l, j, 0] * g_square[l, n, j] * P[l, j, 0] /
                            ((In[l, j, 0] + N0) * (In[l, j, 0] + N0 + g_diag[l, j, 0] * P[l, j, 0]))
                    )
    return -1.0 * gradients_second


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


def multi_wireless_loop(N, L, T, g, lr, P, beta):
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
    P_record = np.zeros((T, L, N, 1))
    global_objective = np.zeros((T, L))
    gradients_record = np.zeros((T, L, N, 1))
    # Prepare g to calculation
    g_diag = np.diagonal(g_square, axis1=1, axis2=2)
    g_diag = g_diag.reshape(L, N, 1)
    g_colum = np.transpose(g_square, axes=(0, 2, 1))
    # Initialize gradients array
    gradients_first = np.zeros((L, N, 1))
    gradients_second = np.zeros((L, N, 1))
    for t in range(T):
        # calculate instance
        In = np.matmul(g_colum, P) - g_diag * P

        # calculate gradients
        numerator = (g_diag / (In + N0))
        gradients_first = (numerator / (1 + numerator * P)) - beta
        gradients_second = calc_second_gradient(N, gradients_second, g_diag, g_square, P, In)

        # update agent vector(P)
        P += lr[t] * (gradients_first + gradients_second)
        # Project the action to [Border_floor, Border_ceil] (Normalization)
        P = np.minimum(np.maximum(P, Border_floor), Border_ceil)

        # Save results in record
        P_record[t] = P
        gradients_record[t] = (gradients_first + gradients_second)

        # Calculate global objective
        temp = np.log(1 + numerator * P)
        global_objective[t] = np.sum(temp[0, :, 0])

    # Finally Let's mean for all L trials
    return P_record[:, 0, :, 0], global_objective[:, 0]

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


######################################## Main Program #############################################################################

# Initialize Parameters for Deep Network
file_path_weights = os.path.join("trains_record", "Wireless_example", "Wireless.pth")
N0 = 0.001
L = 1
L_games = 2000
T = 80000
learning_rate_1 = 0.03 * np.reciprocal(np.power(range(1, T + 1), 0.6))
learning_rate_2 = 0.03 * np.reciprocal(np.power(range(1, T + 1), 0.9))
N = 5
alpha = 10e-2
input_size = 4
output_size = 1
model = Wireless(input_size, output_size)  # Initialize the model with the same architecture
model.load_state_dict(torch.load(file_path_weights, map_location='cpu'))

P_naive_mean = np.zeros((L_games, T, N))
P_net_mean = np.zeros((L_games, T, N))
P_optimal = np.zeros((L_games, N))
global_naive_mean = np.zeros((L_games, T))
global_net_mean = np.zeros((L_games, T))
global_optim = np.zeros((L_games, ))

# Generate gain matrix
for l in range(L_games):
    g = generate_gain_channel(L, N, alpha)

    # Build X_train
    P_init = 0.001 * np.random.rand(L, N, 1)  # Generate Power from uniform distributed
    g_square = g ** 2
    g_diag = np.diagonal(g_square, axis1=1, axis2=2)
    g_diag = g_diag.reshape(L, N, 1)
    g_colum = np.transpose(g_square, axes=(0, 2, 1))
    In_init = np.matmul(g_colum, P_init) - g_diag * P_init

    # Calcalute optimal results
    # Define bounds for P, where each P_n should be between 0 and 1 (Optimization constraint)
    bounds = [(0, 1)] * N  # Create a list of N tuples, each with bounds (0, 1)
    # result = optim.minimize(objective_function, P, args=(g_diag, In, N0), bounds=[(0, 1)] * N)
    result = optim.differential_evolution(objective_function, bounds, args=(g_diag, np.squeeze(g_colum, axis=0), N0),
                                          maxiter=10000)

    # Extract the optimal power allocation from the result
    optimal_power_allocation = result.x
    optimal_objective_value = -result.fun

    # Build Input
    X_test = np.zeros((N, 4))
    In = np.matmul(g_colum, np.ones((L, N, 1))) - g_diag * np.ones((L, N, 1))
    X_test[:, 0] = np.log(g_diag[0, :, 0])
    X_test[:, 1] = np.log(In[0, :, 0])
    X_test[:, 2] = np.log(1 + (g_diag[0, :, 0] / (In[0, :, 0] + N0)))
    kernel = np.exp(- np.abs(In[0, :, 0] - g_diag[0, :, 0]) / 2)
    X_test[:, 3] = kernel

    # Take beta results
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        X_test = torch.tensor(X_test, dtype=torch.float32)
        # Squeeze the input tensor to match the Fc size
        X_test = X_test.squeeze(dim=-1)
        outputs = model(X_test)
        outputs = outputs.squeeze(dim=-1)
        beta = outputs.numpy()


    print(f"l = {l}   Beta = {beta}")

    # # Reshape beta fot correspond to train size in gradient
    beta = beta.reshape(1, N, 1)
    P_naive, global_naive = multi_wireless_loop(N, L, T, g, learning_rate_1, P_init, beta=0)
    P_network, global_network = multi_wireless_loop(N, L, T, g, learning_rate_2, P_init, beta=beta)

    P_naive_mean[l] = P_naive
    P_net_mean[l] = P_network
    P_optimal[l] = optimal_power_allocation
    global_naive_mean[l] = global_naive
    global_net_mean[l] = global_network
    global_optim[l] = optimal_objective_value

P_naive_mean = np.mean(P_naive_mean, axis=0)
P_net_mean = np.mean(P_net_mean, axis=0)
P_optimal = np.mean(P_optimal, axis=0)
global_naive_mean = np.sum(global_naive_mean, axis=0)
global_net_mean = np.sum(global_net_mean, axis=0)
global_optim = np.sum(global_optim, axis=0)
# Plot results
t = np.arange(T)
plt.figure(1)
plt.subplot(1, 3, 1)
for n in range(N):
    Pn = P_naive_mean[:, n]
    plt.plot(t, Pn, label=f"P{n}"), plt.xlabel("# Iteration"),
    plt.ylabel("# candidate action"), plt.legend()
    plt.title("Naive")
plt.subplot(1, 3, 2)
for n in range(N):
    Pn = P_net_mean[:, n]
    plt.plot(t, Pn, label=f"P{n}"), plt.xlabel("# Iteration"),
    plt.ylabel("# candidate action"), plt.legend()
    plt.title("Network")
plt.subplot(1, 3, 3)
for n in range(N):
    Pn = np.ones((T, )) * P_optimal[n]
    plt.plot(t, Pn, label=f"P{n}"), plt.xlabel("# Iteration"),
    plt.ylabel("# candidate action"), plt.legend()
    plt.title("Optimization Package solution")

plt.figure(2)
plt.title("Naive Vs Beta Network")
plt.plot(t, global_naive_mean, label='naive'),
plt.plot(t, global_net_mean, label='NN'), \
plt.plot(t, np.ones(T, ) * global_optim, '--k', label='Optimal package')
plt.xlabel("# Iteration"), plt.ylabel("Global Objective"), plt.title("Compare Global"), plt.legend()

plt.figure(3)
plt.title("Total Power")
plt.plot(t, np.ones(T, ) * np.sum(P_naive[T-1]), label='naive'),
plt.plot(t, np.ones(T, ) * np.sum(P_network[T-1]), label='NN'),
plt.plot(t, np.ones(T, ) * np.sum(optimal_power_allocation), '--k', label='Optimal package'),
plt.xlabel("# Iteration"), plt.legend()
plt.show()

print("Finsh !!!")




# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import os
# import torch.nn.init as init
# from numba import jit
#
# # %% Setting Archticture network
# class Wireless(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(Wireless, self).__init__()
#         self.fc0 = nn.Linear(in_features=input_size, out_features=4096)
#         self.bn0 = nn.BatchNorm1d(4096)
#         self.fc1 = nn.Linear(in_features=4096, out_features=2048)
#         self.bn1 = nn.BatchNorm1d(2048)
#         self.fc2 = nn.Linear(2048, 1024)
#         self.bn2 = nn.BatchNorm1d(1024)
#         self.fc3 = nn.Linear(1024, 512)
#         self.bn3 = nn.BatchNorm1d(512)
#         self.fc4 = nn.Linear(512, 256)
#         self.bn4 = nn.BatchNorm1d(256)
#         self.fc5 = nn.Linear(256, 128)
#         self.bn5 = nn.BatchNorm1d(128)
#         self.fc6 = nn.Linear(128, 64)
#         self.bn6 = nn.BatchNorm1d(64)
#         self.fc7 = nn.Linear(64, 32)
#         self.bn7 = nn.BatchNorm1d(32)
#         self.fc8 = nn.Linear(32, output_size)
#         # Initialize the weights using Kaiming (He) Normal initialization for ReLU
#         self.init_weights()
#
#     def forward(self, x):
#         # Main path
#         x = F.relu(self.bn0(self.fc0(x)))
#         x = F.relu(self.bn1(self.fc1(x)))
#         x = F.relu(self.bn2(self.fc2(x)))
#         x = F.relu(self.bn3(self.fc3(x)))
#         x = F.relu(self.bn4(self.fc4(x)))
#         x = F.relu(self.bn5(self.fc5(x)))
#         x = F.relu(self.bn6(self.fc6(x)))
#         x = F.relu(self.bn7(self.fc7(x)))
#         x = F.relu(self.fc8(x))
#         return x
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 # init.normal_(m.weight, mean=0, std=1.0)
#                 init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#
# @jit(nopython=True)
# def calc_second_gradient(N, gradients_second, g_diag, g_square, P, In):
#     """
#     :param N: int number of Players
#     :param gradients_second: zero numpy array size (L, N, 1)
#     :param g_diag: numpy array size (L, N, 1)
#     :param g_square: numpy array size (L, N, N)
#     :param P: numpy array size (L, N, 1)
#     :param In: numpy array size (L, N, 1)
#     :return: gradients_second: numpy array size (L, N, 1)
#     """
#     N0 = 0.001
#     L, _, _ = gradients_second.shape
#     for n in range(N):
#         for j in range(N):
#             if n != j:
#                 for l in range(L):
#                     gradients_second[l, n, 0] += (
#                         g_diag[l, j, 0] * g_square[l, n, j] * P[l, j, 0] /
#                         ((In[l, j, 0] + N0) * (In[l, j, 0] + N0 + g_diag[l, j, 0] * P[l, j, 0]))
#                     )
#     return -1.0 * gradients_second
#
#
# def generate_gain_channel(L, N, alpha):
#     """
#     :param L: (int) trials of game
#     :param N:  (int) number of players
#     :param alpha:  (int) some constant
#     :return: g return the gain of channels between players
#     """
#     # Generate random transceiver locations in a radius of 1
#     Transreceivers = np.random.rand(L, N, 2) * 2 - 1  # Scale to (-1, 1) and then to (-1, 1) radius 1
#     # Generate random receiver locations in a radius of 0.1 around each transceiver
#     Rlink = 0.1
#     ReceiverOffsets = Rlink * (np.random.rand(L, N, 2) * 2 - 1)  # Scale to (-1, 1) and then to (-0.1, 0.1)
#     Receivers = Transreceivers + ReceiverOffsets
#     # Calculate distances between transceivers and receivers
#     distances = np.linalg.norm(Transreceivers[:, :, np.newaxis, :] - Receivers[:, np.newaxis, :, :], axis=3)
#     g = alpha / (distances ** 2)
#     return g
#
# def multi_wireless_loop(N, L, T, g, lr, P, beta):
#     """
#     :param N: (int) number of players
#     :param L: (int) trials of game
#     :param T: (int) duration of experiment
#     :param g: (L, N, N) channel gain matrix
#     :param lr: (T, ) learning rate vector
#     :return: mean_P_record (T, N) the P array with record on time
#     """
#     # Initialize Parameters
#     Border_floor = 0
#     Border_ceil = 1
#     N0 = 0.001
#     g_square = g ** 2
#     # Initialize record variables
#     P_record = np.zeros((T, L, N, 1))
#     global_objective = np.zeros((T, L))
#     gradients_record = np.zeros((T, L, N, 1))
#     # Prepare g to calculation
#     g_diag = np.diagonal(g_square, axis1=1, axis2=2)
#     g_diag = g_diag.reshape(L, N, 1)
#     g_colum = np.transpose(g_square, axes=(0, 2, 1))
#     # Initialize gradients array
#     gradients_first = np.zeros((L, N, 1))
#     gradients_second = np.zeros((L, N, 1))
#     for t in range(T):
#         # calculate instance
#         In = np.matmul(g_colum, P) - g_diag * P
#
#         # calculate gradients
#         numerator = (g_diag / (In + N0))
#         gradients_first = (numerator / (1 + numerator * P)) - beta
#         gradients_second = calc_second_gradient(N, gradients_second, g_diag, g_square, P, In)
#
#         # update agent vector(P)
#         P += lr[t] * (gradients_first + gradients_second)
#         # Project the action to [Border_floor, Border_ceil] (Normalization)
#         P = np.minimum(np.maximum(P, Border_floor), Border_ceil)
#
#         # Save results in record
#         P_record[t] = P
#         gradients_record[t] = gradients_first + gradients_second
#
#         # Calculate global objective
#         temp = np.log(1 + numerator * P)
#         temp = temp.squeeze()
#         global_objective[t] = np.sum(temp, axis=1)
#
#     # Finally Let's mean for all L trials
#     P_record = P_record.squeeze()
#     gradients_record = gradients_record.squeeze()
#     mean_P_record = np.mean(P_record, axis=1)
#     mean_gradients_record = np.mean(gradients_record, axis=1)
#     mean_global_objective = np.mean(global_objective, axis=1)
#     return mean_P_record, mean_global_objective
#
#
# ######################################## Main Program #############################################################################
#
# # Initialize Parameters for Deep Network
# file_path_weights = os.path.join("trains_record", "Wireless_example", "Wireless.pth")
# L = 2000
# T = 60000
# learning_rate = 0.03 * np.reciprocal(np.power(range(1, T + 1), 0.65))
# N = 10
# alpha = 10e-2
# input_size = 10 * N
# output_size = N  # Size of output (vectorized Q matrix)
# model = Wireless(input_size, output_size)  # Initialize the model with the same architecture
# model.load_state_dict(torch.load(file_path_weights, map_location='cpu'))
#
#
# # Generate gain matrix
# g = generate_gain_channel(L, N, alpha)
#
# # Build X_train
# P_init = 0.1 * np.random.rand(L, N, 1)  # Generate Power from uniform distributed
# g_square = g ** 2
# g_diag = np.diagonal(g_square, axis1=1, axis2=2)
# g_diag = g_diag.reshape(L, N, 1)
# g_colum = np.transpose(g_square, axes=(0, 2, 1))
# In_init = np.matmul(g_colum, P_init) - g_diag * P_init
# X_test = np.zeros((L, 10 * N))
#
# # Take beta results
# model.eval()  # Set the model to evaluation mode
# with torch.no_grad():
#     X_test = torch.tensor(X_test, dtype=torch.float32)
#     # Squeeze the input tensor to match the Fc size
#     X_test = X_test.squeeze(dim=-1)
#     outputs = model(X_test)
#     outputs = outputs.squeeze(dim=-1)
#     beta = outputs.numpy()
#
# # Reshape beta fot correspond to train size in gradient
# beta = beta[:, np.newaxis, np.newaxis] * np.ones((L, N, 1))
#
# P_naive, global_naive = multi_wireless_loop(N, L, T, g, learning_rate, P_init, beta=0)
# P_network, global_network = multi_wireless_loop(N, L, T, g, learning_rate, P_init, beta=beta)
#
#
# # Plot results
# t = np.arange(T)
# plt.figure(1)
# plt.subplot(1, 2, 1)
# for n in range(N):
#     Pn = P_naive[:, n]
#     plt.plot(t, Pn, label=f"P{n}"), plt.xlabel("# Iteration"),
#     plt.ylabel("# candidate action"), plt.legend()
#     plt.title("Naive")
# plt.subplot(1, 2, 2)
# for n in range(N):
#     Pn = P_network[:, n]
#     plt.plot(t, Pn, label=f"P{n}"), plt.xlabel("# Iteration"),
#     plt.ylabel("# candidate action"), plt.legend()
#     plt.title("Network")
#
# plt.figure(2)
# plt.title("Naive Vs Beta Network")
# plt.subplot(1, 2, 1)
# plt.plot(t, global_naive), plt.xlabel("# Iteration"), plt.ylabel("Global Objective"), plt.title("Naive")
# plt.subplot(1, 2, 2)
# plt.plot(t, global_network), plt.xlabel("# Iteration"), plt.ylabel("Global Objective"), plt.title("Network")
# plt.show()
#
# print("Finsh !!!")