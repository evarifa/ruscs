import numpy as np
import networkx as nx

def initialize_network_RRG_list(num_nodes, degree): # 1-based index
    G = nx.random_regular_graph(d = degree, n = num_nodes)
    neighbors = [(np.array(G[i]) + 1).tolist() for i in range(num_nodes)]
    degree_tot = [len(neighbors[i]) for i in range(len(neighbors))]
    kmax = max(degree_tot)
    return neighbors, kmax