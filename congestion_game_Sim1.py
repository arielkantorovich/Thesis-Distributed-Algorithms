"""
Created on : ------

@author: Ariel_Kantorovich
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

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

def Generate_Game_Param(K):
    """
    :param K:(int), Number of strategy in game
    :return:
    """
    A = np.random.rand(K,)
    B = np.random.rand(K,)
    C = np.random.rand(K,)
    return A, B, C

def Initialize_action(N, P):
    """
    This function build action Matrix each player how much budget is spend on path Pi
    :param N: (int) number of players
    :param P: (int) number of paths from source to targer
    :return: Xe (2D-array) size NxP
    """
    Xn_k = np.random.rand(N, P)
    row_sums = np.sum(Xn_k, axis=1, keepdims=True)
    Xn_k /= row_sums
    return Xn_k

def calc_gradient(C_k, Xn_k, B, C, Xe, Nash_flag=False):
    """
    :param C_k: size (K, )
    :param Xn_k: size (N, K)
    :param B: size (K,)
    :param C: size (K,)
    :param Nash_flag: bool, specify if calculate only gradient Nash
    :return: gradient of each player size (N, )
    """
    self_gradient = C_k * (Xn_k > 0) + Xn_k * (B + 2 * C * Xe)
    if Nash_flag:
        return self_gradient
    global_grad = C_k * (Xn_k > 0) + (B + 2 * C * Xe) * np.sum(Xn_k, axis=0, keepdims=True) * (Xn_k > 0)
    return global_grad


def project_onto_simplex(V, z=1):
    """
    Project matrix X onto the probability simplex.
    :param Y: numpy array, matrix of size NxK where D=K
    :param z: integer, norm of projected vector
    :return X: numpy array, matrix of the same size as X, projected onto the probability simplex
    """
    n_features = V.shape[1]
    U = np.sort(V, axis=1)[:, ::-1]
    z = np.ones(len(V)) * z
    cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
    ind = np.arange(n_features) + 1
    cond = U - cssv / ind > 0
    rho = np.count_nonzero(cond, axis=1)
    theta = cssv[np.arange(len(V)), rho - 1] / rho
    return np.maximum(V - theta[:, np.newaxis], 0)

# Define Seed for debug
np.random.seed(0)

# Hyper Parameters
prob = 0.1
nodes = 20
N = 600 # Number of players
T = 5000
lr_global = 0.03 * np.reciprocal(np.power(range(1, T + 1), 0.65))
lr_nash = 0.03 * np.reciprocal(np.power(range(1, T + 1), 0.65))
# lr = 0.03 * np.reciprocal(np.power(range(1, T + 1), 0.6))
K = 20 # Number of strategy

# Record Variables
nash_Cn_t = np.zeros((T, N)) # record cost of players
global_Cn_t = np.zeros((T, N)) # record cost of players
global_grad_t = np.zeros((T, N, K)) # record gradient
nash_grad_t = np.zeros((T, N, K)) # record gradient
global_action_record = np.zeros((T, N, K)) # Record action of players
nash_action_record = np.zeros((T, N, K))
global_record = np.zeros((T, ))
nash_record = np.zeros((T, ))
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
A, B, C = Generate_Game_Param(K)

# Generate action of players
nash_Xn_k = Initialize_action(N, K)
global_Xn_k = nash_Xn_k.copy()


# Main loop
for t in range(T):
    # Calculate load
    Xe_nash = np.sum(nash_Xn_k, axis=0)
    Xe_global = np.sum(global_Xn_k, axis=0)

    # calculate congestion
    C_k_nash = A + B * Xe_nash + C * (Xe_nash ** 2)
    C_k_global = A + B * Xe_global + C * (Xe_global ** 2)

    # calculate cost of player n
    C_n_nash = np.sum(nash_Xn_k * C_k_nash, axis=1)
    C_n_global = np.sum(global_Xn_k * C_k_global, axis=1)
    nash_Cn_t[t] = C_n_nash
    global_Cn_t[t] = C_n_global

    # Calculate Global & Nash record
    global_record[t] = np.sum(C_n_global)
    nash_record[t] = np.sum(C_n_nash)

    # calculate gradient
    grad_nash = calc_gradient(C_k_nash, nash_Xn_k, B, C, Xe_nash, Nash_flag=True)
    grad_global = calc_gradient(C_k_global, global_Xn_k, B, C, Xe_global, Nash_flag=False)
    global_grad_t[t] = grad_global
    nash_grad_t[t] = grad_nash

    # Update action of players
    nash_Xn_k = nash_Xn_k - lr_nash[t] * grad_nash
    global_Xn_k = global_Xn_k - lr_global[t] * grad_global
    global_action_record[t] = global_Xn_k
    nash_action_record[t] = nash_Xn_k

    # Project gradient to constraint
    nash_Xn_k = project_onto_simplex(nash_Xn_k)
    global_Xn_k = project_onto_simplex(global_Xn_k)


# Plot results
t = np.arange(T)
plt.figure(1)
plt.plot(t, global_record, label="Global")
plt.plot(t, nash_record, label="Nash")
plt.xlabel("# Iterations"), plt.ylabel("# Score"), plt.legend()
plt.show()

print("Finsh")