"""
Created on : ------

@author: Ariel_Kantorovich
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import torch.nn.init as init
import os
import torch.nn.functional as F


class CongestionCNN(nn.Module):
    def __init__(self, input_channels=1, output_size=4):
        super(CongestionCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(3, 3), padding=1)
        self.bn_conv1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))
        self.bn_conv2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 1 * 2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 16)
        self.bn4 = nn.BatchNorm1d(16)
        self.fc5 = nn.Linear(16, output_size)

    def forward(self, x):
        x = self.relu(self.bn_conv1(self.conv1(x)))
        x = self.relu(self.bn_conv2(self.conv2(x)))
        x = self.flatten(x)
        # Fully connected
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.relu(self.fc5(x))
        return x
class Congestion_AutoEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Congestion_AutoEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.fc6 = nn.Linear(32, output_size)
        self.init_weights()

    def forward(self, x):
        # Encoder Part
        x1 = torch.relu(self.bn1(self.fc1(x)))
        x2 = torch.relu(self.bn2(self.fc2(x1)))
        x3 = torch.relu(self.bn3(self.fc3(x2)))
        # Decoder Part
        x4 = torch.relu(self.bn4(self.fc4(x3))) + x2
        x5 = torch.relu(self.bn5(self.fc5(x4))) + x1

        # Final layer without activation for regression
        output = self.fc6(x5)
        return output

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

class Congestion(nn.Module):
    def __init__(self, input_size, output_size):
        super(Congestion, self).__init__()
        self.fc0 = nn.Linear(input_size, 128)
        self.bn0 = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(128, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, output_size)
        self.init_weights()

    def forward(self, x):
        x = F.relu(self.bn0(self.fc0(x)))
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.fc5(x))
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

def Draw_Graph(G, draw_edges=False):
    """
    The function drawing unDirected graph with edges
    :param G:
    :return: None
    """
    node_colors = ["red" if node == "S" else "green" if node == "T" else "skyblue" for node in G.nodes]
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color=node_colors)

    if draw_edges:
        edge_labels = nx.get_edge_attributes(G, 'weight')
        for edge, weight in edge_labels.items():
            x, y = pos[edge[0]]
            x_, y_ = pos[edge[1]]
            mid_x, mid_y = (x + x_) / 2, (y + y_) / 2
            plt.text(mid_x, mid_y, s=f'{weight:.2f}', bbox=dict(facecolor='white', alpha=0.5), ha='center', va='center')
    plt.show()

def Generate_graph(num_nodes=60, p=0.1):
    """
    THis function generate Erdős–Rényi graph model
    :param num_nodes: (int) number of nodes in graph
    :param p: (int), prob
    :return: G: (object),  graph
    """
    G = nx.erdos_renyi_graph(num_nodes, p)
    G = nx.relabel_nodes(G, {0: "S", len(G) - 1: "T"})
    return G

def find_all_paths(graph, source, target):
    """
    :param graph: (object)
    :param source: (str)
    :param target: (str)
    :return:
    """
    return list(nx.all_simple_paths(graph, source=source, target=target))

def Generate_Game_Param(L, K):
    """
    :param K:(int), Number of strategy in game
    :return:
    """
    A = np.random.rand(L, K,)
    B = np.random.rand(L, K,)
    C = np.random.rand(L, K,)
    return A, B, C

def Initialize_action(L, N, P):
    """
    This function build action Matrix each player how much budget is spend on path Pi
    :param N: (int) number of players
    :param P: (int) number of paths from source to targer
    :return: Xe (2D-array) size NxP
    """
    Xn_k = np.random.rand(L, N, P)
    row_sums = np.sum(Xn_k, axis=2, keepdims=True)
    Xn_k /= row_sums
    return Xn_k

def calc_gradient(C_k, Xn_k, B, C, Xe, Nash_flag=False, beta=0, gamma=0):
    """
    :param C_k: size (L, K, )
    :param Xn_k: size (L, N, K)
    :param B: size (L, K,)
    :param C: size (L, K,)
    :param Nash_flag: bool, specify if calculate only gradient Nash
    :param beta: size (L, K,)
    :param gamma: size (L, K,)
    :return: gradient of each player size (N, )
    """
    N = Xn_k.shape[1]
    if Nash_flag:
        self_gradient = C_k * (Xn_k > 0) + Xn_k * np.repeat((B + 2 * C * Xe)[:, np.newaxis, :], N, axis=1) + beta * (Xn_k > 0)
        return self_gradient
    sum_Xn_k_expanded = np.repeat(np.sum(Xn_k, axis=1)[:, np.newaxis, :], N, axis=1)
    global_grad = C_k * (Xn_k > 0) + np.repeat((B + 2 * C * Xe)[:, np.newaxis, :], N, axis=1) * sum_Xn_k_expanded * (Xn_k > 0)
    return global_grad


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

def DeBug_NN(B, C, Xe_global, global_Xn_k, nash_const_Xn_k, Ck=0):
    """
    :param B:
    :param C:
    :param Xe_global:
    :param global_Xn_k:
    :param nash_const_Xn_k:
    :return:
    """
    beta = np.repeat((B + 2 * C * Xe_global)[:, np.newaxis, :], N, axis=1) * global_Xn_k + np.repeat(Ck[:, np.newaxis, :], N, axis=1)
    beta = beta * -1.0
    # beta = np.repeat((B + 2 * C * Xe_global)[:, np.newaxis, :], N, axis=1) * (
    #             np.repeat(Xe_global[:, np.newaxis, :], N, axis=1) - global_Xn_k)  # Optimal Check
    for t in range(T):
        # Calculate Load
        Xe_nash_const = np.sum(nash_const_Xn_k, axis=1)
        # Calculate congestion
        C_k_nash_const = A + B * Xe_nash_const + C * (Xe_nash_const ** 2)
        # Expand Ck
        C_k_nash_const_expanded = np.repeat(C_k_nash_const[:, np.newaxis, :], N, axis=1)
        # Calculate cost of player n
        C_n_nash_const = np.sum(nash_const_Xn_k * C_k_nash_const_expanded, axis=2)
        # Record results
        nash_const_record[t] = np.sum(C_n_nash_const, axis=1)
        # calculate gradient
        grad_nash_const = calc_gradient(C_k_nash_const_expanded, nash_const_Xn_k, B, C, Xe_nash_const, Nash_flag=True,
                                        beta=beta, gamma=0)
        # Update action of players
        nash_const_Xn_k = nash_const_Xn_k - lr_nash_const[t] * grad_nash_const
        # Project gradient to constraint
        nash_const_Xn_k = project_onto_simplex(nash_const_Xn_k)
    return nash_const_record

def EST_ABS(Ck, Xe, sigma=2):
    """
    :param Ck: ndarray - (L, K)
    :param Xe: ndarray - (L, K)
    :param sigma: (int)
    :return: A, B, C Estimation (L, N, K)
    """
    X_e_K = np.concatenate([np.ones_like(Xe).reshape(L, K, 1), Xe.reshape(L, K, 1), (Xe ** 2).reshape(L, K, 1)],axis=-1)
    SVD_est = np.linalg.pinv(X_e_K)
    ABC = np.matmul(SVD_est, Ck.reshape(L, K, 1))
    # Extract A, B, C Psudo inverse
    A = ABC[:, 0].squeeze(axis=-1)
    B = ABC[:, 1].squeeze(axis=-1)
    C = ABC[:, 2].squeeze(axis=-1)
    # Expand such can add noise
    A = np.repeat(A[:, np.newaxis], K, axis=-1) + np.random.randn(L, K) * sigma
    B = np.repeat(B[:, np.newaxis], K, axis=-1) + np.random.randn(L, K) * sigma
    C = np.repeat(C[:, np.newaxis], K, axis=-1) + np.random.randn(L, K) * sigma
    # Expand such will coresspond to N players
    A = np.repeat(A[:, np.newaxis, :], N, axis=1)
    B = np.repeat(B[:, np.newaxis, :], N, axis=1)
    C = np.repeat(C[:, np.newaxis, :], N, axis=1)
    return A, B, C


# Define Seed for debug
np.random.seed(0)

# Hyper Parameters
prob = 0.1
nodes = 20
N = 20 # Number of players
T = 500
L = 200
K = 4 # Number of strategy
lr_global = 0.03 * np.reciprocal(np.power(range(1, T + 1), 0.75))
lr_nash = 0.03 * np.reciprocal(np.power(range(1, T + 1), 0.75))
lr_nash_const = 0.03 * np.reciprocal(np.power(range(1, T + 1), 0.75))
Debug_flag = False
Generate_Train = False
Normalize_flag = False

# Record Variables
if Debug_flag:
    nash_Cn_t = np.zeros((T, L, N), dtype=np.float32)  # record cost of players
    global_Cn_t = np.zeros((T, L, N), dtype=np.float32)  # record cost of players
    global_grad_t = np.zeros((T, L, N, K), dtype=np.float32)  # record gradient
    nash_grad_t = np.zeros((T, L, N, K), dtype=np.float32)  # record gradient
    global_action_record = np.zeros((T, L, N, K), dtype=np.float32)  # Record action of players
    nash_action_record = np.zeros((T, L, N, K), dtype=np.float32)

# Record Final Cost
global_record = np.zeros((T, L), dtype=np.float32)
nash_record = np.zeros((T, L), dtype=np.float32)
nash_const_record = np.zeros((T, L), dtype=np.float32)
nash_tax_record = np.zeros((T, L), dtype=np.float32)
#############################################################################
# # Generate Graph
# G = Generate_graph(nodes, prob)
#
# # Checking connected graph
# while not(nx.is_connected(G)):
#     G = Generate_graph(nodes, prob)
#
# # Draw Graph
# Draw_Graph(G, draw_edges=False)
#
# # Find all paths from source (node 0) to target (node N-1)
# paths = find_all_paths(G, source="S", target="T")
# K = len(paths) # Number of strategy

# # Extract Graph as adjacent Matrix
# adj_matrix = nx.to_numpy_array(G, dtype=float)
#####################################################################################################
# Generate A, B, C Parameters
A, B, C = Generate_Game_Param(L, K)


# Generate action of players
nash_Xn_k = Initialize_action(L, N, K)
global_Xn_k = nash_Xn_k.copy()
nash_const_Xn_k = nash_Xn_k.copy()
nash_tax_Xn_k = nash_Xn_k.copy()

# Prepare data to Network
Xe_nash = np.sum(nash_Xn_k, axis=1)
Xe_nash_expand = np.repeat(Xe_nash[:, np.newaxis, :], N, axis=1)
C_k_nash = A + B * Xe_nash + C * (Xe_nash ** 2)
C_k_nash_expanded = np.repeat(C_k_nash[:, np.newaxis, :], N, axis=1)

# Save Data for ML training
if Generate_Train:
    # Save arrays
    np.save("Numpy_array_save/Xn_k.npy", nash_Xn_k)
    np.save("Numpy_array_save/Xe.npy", Xe_nash_expand)
    np.save("Numpy_array_save/Ck.npy", C_k_nash_expanded)

# Prepare Data to Network
# A_G, B_G, C_G = EST_ABS(C_k_nash, Xe_nash, sigma=2)
ONE = np.ones_like(Xe_nash_expand)
X_test = np.concatenate([nash_Xn_k, C_k_nash_expanded, ONE, Xe_nash_expand, Xe_nash_expand ** 2], axis=-1)
X_test = np.reshape(X_test, (L * N, 5 * K))
# X_test = np.concatenate((nash_Xn_k, Xe_nash_expand, C_k_nash_expanded), axis=2)
# X_test = X_test.reshape(N*L, 1, K, 3)

if Normalize_flag:
    file_path_mean = os.path.join("Numpy_array_save", "X_mean(L=20k).npy")
    X_mean = np.load(file_path_mean)
    file_path_std = os.path.join("Numpy_array_save", "X_std(L=20k).npy")
    X_std = np.load(file_path_std)
    # Calculate Zero - score normalization
    X_test = (X_test - X_mean) / X_std

# Get Beta from Neural Network
device = ["cuda" if torch.cuda.is_available() else "cpu"]
file_path_weights = os.path.join("trains_record", "Congestion_game_with_Adam", "Congestion(Relu_SGD).pth")
input_size = 5 * K
output_size = K
model = Congestion(input_size=input_size, output_size=output_size)  # Initialize the model with the same architecture
model.load_state_dict(torch.load(file_path_weights, map_location='cpu'))
# Take beta results
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    X_test = torch.tensor(X_test, dtype=torch.float32)
    # Squeeze the input tensor to match the Fc size
    X_test = X_test.squeeze(dim=-1)
    outputs = model(X_test)
    outputs = outputs.squeeze(dim=-1)
    beta = outputs.numpy()
# Change Beta such will be in size L, N, K
beta = - 1.0 * np.reshape(beta, (L, N, K))

# Main loop
for t in range(T):
    # Calculate load
    Xe_nash = np.sum(nash_Xn_k, axis=1)
    Xe_global = np.sum(global_Xn_k, axis=1)
    Xe_nash_const = np.sum(nash_const_Xn_k, axis=1)
    Xe_nash_tax = np.sum(nash_tax_Xn_k, axis=1)

    # calculate congestion
    C_k_nash = A + B * Xe_nash + C * (Xe_nash ** 2)
    C_k_global = A + B * Xe_global + C * (Xe_global ** 2)
    C_k_nash_const = A + B * Xe_nash_const + C * (Xe_nash_const ** 2)
    C_k_nash_tax = A + B * Xe_nash_tax + C * (Xe_nash_tax ** 2)

    # Expand C_k such is corresponded to handle with L games parallel
    C_k_nash_expanded = np.repeat(C_k_nash[:, np.newaxis, :], N, axis=1)
    C_k_global_expanded = np.repeat(C_k_global[:, np.newaxis, :], N, axis=1)
    C_k_nash_const_expanded = np.repeat(C_k_nash_const[:, np.newaxis, :], N, axis=1)
    C_k_nash_tax_expanded = np.repeat(C_k_nash_tax[:, np.newaxis, :], N, axis=1)

    # calculate cost of player n
    C_n_nash = np.sum(nash_Xn_k * C_k_nash_expanded, axis=2)
    C_n_global = np.sum(global_Xn_k * C_k_global_expanded, axis=2)
    C_n_nash_const = np.sum(nash_const_Xn_k * C_k_nash_const_expanded, axis=2)
    C_n_nash_tax = np.sum(nash_tax_Xn_k * C_k_nash_tax_expanded, axis=2)
    # nash_Cn_t[t] = C_n_nash
    # global_Cn_t[t] = C_n_global

    # Calculate Global & Nash record
    global_record[t] = np.sum(C_n_global, axis=1)
    nash_record[t] = np.sum(C_n_nash, axis=1)
    nash_const_record[t] = np.sum(C_n_nash_const, axis=1)
    nash_tax_record[t] = np.sum(C_n_nash_tax, axis=1)


    # calculate gradient
    grad_nash = calc_gradient(C_k_nash_expanded, nash_Xn_k, B, C, Xe_nash, Nash_flag=True)
    grad_global = calc_gradient(C_k_global_expanded, global_Xn_k, B, C, Xe_global, Nash_flag=False)
    grad_nash_const = calc_gradient(C_k_nash_const_expanded, nash_const_Xn_k, B, C, Xe_nash_const, Nash_flag=True, beta=beta, gamma=0) ### This NN
    # Calculate tax grad
    tax_regulizaer = B + 4 * C * Xe_nash_tax
    grad_nash_tax = C_k_nash_tax_expanded * (nash_tax_Xn_k > 0) + nash_tax_Xn_k * np.repeat((B + 2 * C * Xe_nash_tax)[:, np.newaxis, :], N, axis=1) \
                    + np.repeat(tax_regulizaer[:, np.newaxis, :], N, axis=1) * (nash_tax_Xn_k > 0)
    # global_grad_t[t] = grad_global
    # nash_grad_t[t] = grad_nash

    # Update action of players
    nash_Xn_k = nash_Xn_k - lr_nash[t] * grad_nash
    global_Xn_k = global_Xn_k - lr_global[t] * grad_global
    nash_const_Xn_k = nash_const_Xn_k - lr_nash_const[t] * grad_nash_const
    nash_tax_Xn_k = nash_tax_Xn_k - lr_nash[t] * grad_nash_tax
    # global_action_record[t] = global_Xn_k
    # nash_action_record[t] = nash_Xn_k

    # Project gradient to constraint
    nash_Xn_k = project_onto_simplex(nash_Xn_k)
    global_Xn_k = project_onto_simplex(global_Xn_k)
    nash_const_Xn_k = project_onto_simplex(nash_const_Xn_k)
    nash_tax_Xn_k = project_onto_simplex(nash_tax_Xn_k)

if Generate_Train:
    beta = np.repeat((B + 2 * C * Xe_global)[:, np.newaxis, :], N, axis=1) * global_Xn_k + np.repeat(C_k_global[:, np.newaxis, :], N, axis=1)
    # beta = beta * -1.0
    # beta = np.repeat((B + 2 * C * Xe_global)[:, np.newaxis, :], N, axis=1) * (
    #         np.repeat(Xe_global[:, np.newaxis, :], N, axis=1) - global_Xn_k)  # Optimal Check
    np.save("Numpy_array_save/beta.npy", beta)

# nash_const_record = DeBug_NN(B, C, Xe_global, global_Xn_k, nash_const_Xn_k, C_k_global)

# Plot results
t = np.arange(T)
plt.figure(1)
plt.plot(t, np.sum(global_record, axis=1), '--k', label="Global")
plt.plot(t, np.sum(nash_record, axis=1), label="Nash")
plt.plot(t, np.sum(nash_const_record, axis=1), label="NN")
plt.plot(t, np.sum(nash_tax_record, axis=1), label="Tax")
plt.xlabel("# Iterations"), plt.ylabel("# Score"), plt.legend()
plt.show()

print("Finsh")