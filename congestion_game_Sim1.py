"""
Created on : ------

@author: Ariel_Kantorovich
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def Draw_Graph(G):
    """
    The function drawing unDirected graph with edges
    :param G:
    :return: None
    """
    node_colors = ["red" if node == "S" else "green" if node == "T" else "skyblue" for node in G.nodes]
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color=node_colors)
    # edge_labels = nx.get_edge_attributes(G, 'weight')
    # for edge, weight in edge_labels.items():
    #     x, y = pos[edge[0]]
    #     x_, y_ = pos[edge[1]]
    #     mid_x, mid_y = (x + x_) / 2, (y + y_) / 2
    #     plt.text(mid_x, mid_y, s=f'{weight:.2f}', bbox=dict(facecolor='white', alpha=0.5), ha='center', va='center')
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

def Generate_Game_Param(nodes):
    """
    :param nodes:
    :return:
    """
    A = np.random.rand(nodes, nodes)
    np.fill_diagonal(A, 0)
    B = np.random.rand(nodes, nodes)
    np.fill_diagonal(B, 0)
    C = np.random.rand(nodes, nodes)
    np.fill_diagonal(C, 0)
    return A, B, C

def Initialize_action(N, P):
    """
    This function build action Matrix each player how much budget is spend on path Pi
    :param N: (int) number of players
    :param P: (int) number of paths from source to targer
    :return: Xe (2D-array) size NxP
    """
    Xe = np.random.rand(N, P)
    row_sums = np.sum(Xe, axis=1, keepdims=True)
    Xe /= row_sums
    return Xe

# Hyper Parameters
prob = 0.1
nodes = 15
N = 100 # Number of players

# Generate Graph
G = Generate_graph(nodes, prob)

# Checking connected graph
while not(nx.is_connected(G)):
    G = Generate_graph(nodes, prob)

# Draw Graph
Draw_Graph(G)

# Find all paths from source (node 0) to target (node N-1)
paths = find_all_paths(G, source="S", target="T")

# Extract Graph as adjacent Matrix
adj_matrix = nx.to_numpy_array(G, dtype=float)

# Generate A, B, C Parameters
A, B, C = Generate_Game_Param(nodes)

# Generate action of players
Xe = Initialize_action(N, len(paths))


print("Finsh")