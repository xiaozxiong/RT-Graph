import networkx as nx
import matplotlib.pyplot as plt

# Generate a scale-free graph using the Barabási-Albert model
n = 2000000  # Number of nodes
m = 8    # Number of edges to attach from a new node to existing nodes
G = nx.barabasi_albert_graph(n, m)

# nx.write_edgelist(G, "dataset/synthesised/2M_m8.txt", data=False)

# Degree distribution
degrees = [d for _, d in G.degree()]
plt.hist(degrees, bins=range(min(degrees), max(degrees) + 1), alpha=0.75, color='blue', edgecolor='black')
plt.title("Degree Distribution")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.yscale("log")  # Log scale for the frequency

plt.savefig("degree_distribution_2M_m8.png")
