import networkx as nx
import matplotlib.pyplot as plt
import random

# Generate a power-law degree sequence
n = 1000
gamma = 10.5  # Exponent of the power law >= 2
degree_sequence = [int(random.paretovariate(gamma - 1)) for _ in range(n)]
degree_sequence = [d for d in degree_sequence if d > 0]  # Remove zeros
# Adjust the last degree if the sum is odd
if sum(degree_sequence) % 2 != 0:
    degree_sequence[-1] += 1

# Create a graph using the configuration model
G = nx.configuration_model(degree_sequence)
G = nx.Graph(G)  # Remove parallel edges
G.remove_edges_from(nx.selfloop_edges(G))  # Remove self-loops

# Plot the degree distribution
degrees = [G.degree(node) for node in G.nodes()]
plt.hist(degrees, bins=50)
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.title("Degree Distribution")
plt.savefig("output/img/con_degree_10.5.png")
