import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def powerlaw_graph(n, m, p):
    print("n = {}, m = {}, p = {}".format(n, m, p))

    # Generate the power-law cluster graph
    # An undirected graph by default, indexed starting from 0 by default.
    G = nx.powerlaw_cluster_graph(n, m, p)
    
    print("Nodes: " + str(G.number_of_nodes()) + ", Edges: " + str(G.number_of_edges()))
    # Check if the graph is directed
    print("Is the graph directed: ", G.is_directed())  # Should print False

    nx.write_edgelist(G, "powerlaw_graph/n{}_m{}_p{}.txt".format(n, m, p), data=False)

    #TODO: log-log scale degree distribution
    degrees = [d for n, d in G.degree()]
    degree_counts = Counter(degrees)
    degree, count = zip(*degree_counts.items())

    plt.figure(figsize=(8, 6))
    plt.scatter(degree, count, color='b', edgecolor='k', alpha=0.7, label="Degree Distribution")
    plt.xscale("log")  # Log-scale for x-axis
    plt.yscale("log")  # Log-scale for y-axis
    plt.xlabel("Degree (log scale)", fontsize=12)
    plt.ylabel("Number of Nodes (log scale)", fontsize=12)
    plt.title("Log-Log Degree Distribution (m = {}, p = {})".format(m, p), fontsize=14)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    
    fig_name = "tc/output/img/degree_n{}_m{}_p{}.png".format(n, m, p)
    plt.savefig(fig_name)


if __name__ == "__main__":
    n = 2000000  # Number of nodes
    m = 8     # Number of edges to attach from a new node
    # Probability of forming a triangle [0, 1]
    # for p in np.arange(0.1, 1, 0.1):
    #     powerlaw_graph(n, m, p)

    powerlaw_graph(n, m, 0.99)