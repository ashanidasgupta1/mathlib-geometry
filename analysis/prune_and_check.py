import networkx as nx
import json
import numpy as np
import random
import os

# 1. Path Verification
file_path = "mathlib_graph.json"
if not os.path.exists(file_path):
    print(f"ERROR: {file_path} not found in the current directory.")
    print(f"Current directory is: {os.getcwd()}")
    exit()

# 2. Load the Graph
print("Step 1: Loading Mathlib Graph (this may take a minute)...")
with open(file_path, 'r') as f:
    data = json.load(f)

G = nx.Graph()
for entry in data:
    u = entry['name']
    for v in entry['edges']:
        G.add_edge(u, v)

print(f"Successfully loaded {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# 3. Pruning Hubs
print("Step 2: Identifying top 1% hubs (foundational axioms/definitions)...")
degrees = dict(G.degree())
sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
num_to_prune = int(len(sorted_nodes) * 0.01)
hubs = [node for node, deg in sorted_nodes[:num_to_prune]]

print(f"Pruning {num_to_prune} hubs. Top hub was '{sorted_nodes[0][0]}' with {sorted_nodes[0][1]} edges.")
G.remove_nodes_from(hubs)
print(f"Remaining graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

# 4. Curvature Calculation
def estimate_ricci(u, v):
    set_u = set(G.neighbors(u))
    set_v = set(G.neighbors(v))
    union = len(set_u.union(set_v))
    if union == 0: return 0
    intersection = len(set_u.intersection(set_v))
    # Approximation of Ollivier-Ricci
    return (intersection / union) - (1 / (len(set_u) + len(set_v)))

print("Step 3: Sampling 10,000 edges for curvature analysis...")
edges = list(G.edges())
if not edges:
    print("ERROR: No edges remaining after pruning!")
    exit()

sample_size = min(10000, len(edges))
sampled_edges = random.sample(edges, sample_size)
curvatures = []

for i, (u, v) in enumerate(sampled_edges):
    curvatures.append(estimate_ricci(u, v))
    if i % 2000 == 0:
        print(f"  Processed {i}/{sample_size} edges...")

# 5. FINAL RESULTS
mean_k = np.mean(curvatures)
std_k = np.std(curvatures)
neg_percent = len([k for k in curvatures if k < -0.01]) / sample_size * 100

print("\n" + "="*45)
print("FINAL PRUNED MANIFOLD RESULTS")
print("="*45)
print(f"Mean Ricci Curvature (κ): {mean_k:.6f}")
print(f"Standard Deviation:       {std_k:.6f}")
print(f"Hyperbolic Lean (κ < 0):  {neg_percent:.2f}%")
print("="*45)

if mean_k < 0:
    print("RESULT: Removing hubs REVEALED a Hyperbolic structure.")
else:
    print("RESULT: The manifold remains Spherical even without hubs.")