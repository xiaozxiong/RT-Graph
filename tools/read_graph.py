import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

# Plot the degree distribution on a log-log scale
def plot_degree(file_path):
    G = nx.read_edgelist(fig_name, nodetype=int)
    # Compute the degree of each node
    degrees = [d for n, d in G.degree()]

    # Check if the graph is directed
    print("Is the graph directed: ", G.is_directed())  # Should print False

    degree_counts = Counter(degrees)
    degree, count = zip(*degree_counts.items())

    plt.figure(figsize=(8, 6))
    plt.scatter(degree, count, color='b', edgecolor='k', alpha=0.7, label="Degree Distribution")
    plt.xscale("log")  # Log-scale for x-axis
    plt.yscale("log")  # Log-scale for y-axis
    plt.xlabel("Degree (log scale)", fontsize=12)
    plt.ylabel("Number of Nodes (log scale)", fontsize=12)
    plt.title("Log-Log Degree Distribution (p = {})".format(0.1), fontsize=14)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(fig_name) # "output/img/n1000000_m8_p0.1.png"

def add_header(file_path):
    print("file: {}".format(file_path))
    G = nx.read_edgelist(file_path, nodetype=int)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    edges = list(G.edges)

    output_file = file_path + ".header"
    with open(output_file, "w") as f:
        # Write the header: <number of nodes> <number of edges>
        f.write(f"{num_nodes} {num_edges}\n")
    
        # Write the edges
        for edge in edges:
            f.write(f"{edge[0]} {edge[1]}\n")

    print(f"Edge list saved to {output_file} with header.")

if __name__ == "__main__":
    # Read the edge list from a file
    # G = nx.read_edgelist("powerlaw_graph/n1000000_m8_p0.1.txt", nodetype=int)
    add_header("powerlaw_graph/n1000000_m8_p0.9.txt")


   



