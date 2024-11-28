import networkx as nx
from scipy.io import mmwrite
from scipy.sparse import csr_matrix
import argparse

parser = argparse.ArgumentParser(description="Convert edge list of NetworkX to Matrix Market file")

# parameter
parser.add_argument("--input_file", type=str, help="Path of edge list file")
parser.add_argument("--output_file", type=str, help="Path of matrix market file")

# parse parameter
args = parser.parse_args()
print("Edge list: {}".format(args.input_file))
print("Mtx File: {}".format(args.output_file))

# load a graph in NetworkX
G = nx.read_edgelist(args.input_file, nodetype=int)

# Convert the graph to a SciPy sparse adjacency matrix
adj_matrix = nx.adjacency_matrix(G)  # Returns a SciPy sparse matrix (CSR format)

# Save the adjacency matrix in Matrix Market format
mmwrite(args.output_file, adj_matrix)

print("Matrix Market file saved to: {}".format(args.output_file))
