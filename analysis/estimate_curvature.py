import networkx as nx
import json
import numpy as np
import random
import matplotlib.pyplot as plt

# 1. Load the Graph
print("Loading Mathlib Graph...")
with open("mathlib_graph.json", 'r') as f:
    data = json.load(f)

G = nx.Graph()
for entry in data:
    u = entry['name']
    for v in entry['edges']:
        G.add_edge(u, v)

print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

def estimate_ricci_curvature(u, v):
    """
    Approximates local curvature based on neighborhood overlap.
    Positive: Neighbors are highly connected (cliques/spheres).
    Zero: Neighbors form a grid (flat/Euclidean).
    Negative: Neighbors branch away (tree/hyperbolic).
    """
    # Get immediate neighbors
    set_u = set(G.neighbors(u))
    set_v = set(G.neighbors(v))
    
    # Calculate Jaccard-like overlap
    intersection = len(set_u.intersection(set_v))
    union = len(set_u.union(set_v))
    
    if union == 0: return 0
    
    # Formula: Curvature is roughly (Overlap - Expectation)
    # In a perfect tree (hyperbolic), intersection is 0 or 1, union is large.
    # In a flat grid, intersection is higher.
    kappa = (intersection / union) - (1 / (len(set_u) + len(set_v)))
    return kappa

# 2. Sampling the Manifold
print("Sampling 20,000 edges to discover intrinsic geometry...")
edges = list(G.edges())
sample_size = min(20000, len(edges))
sampled_edges = random.sample(edges, sample_size)

curvatures = []
for u, v in sampled_edges:
    curvatures.append(estimate_ricci_curvature(u, v))

# 3. Statistical Analysis
mean_k = np.mean(curvatures)
std_k = np.std(curvatures)
neg_percent = len([k for k in curvatures if k < -0.01]) / sample_size * 100

print("\n" + "="*40)
print("MATHLIB MANIFOLD SIGNATURE")
print("="*40)
print(f"Mean Ricci Curvature (κ): {mean_k:.6f}")
print(f"Standard Deviation:       {std_k:.6f}")
print(f"Hyperbolic Lean (κ < 0):  {neg_percent:.2f}%")
print("-" * 40)

if mean_k < -0.02:
    print("CONCLUSION: Intrinsic geometry is HYPERBOLIC.")
    print("The Poincaré Ball was the correct manifold choice.")
elif mean_k > 0.02:
    print("CONCLUSION: Intrinsic geometry is SPHERICAL.")
    print("The model should likely be trained on a Hypersphere.")
else:
    print("CONCLUSION: Intrinsic geometry is EUCLIDEAN.")
    print("Standard flat embeddings are sufficient.")

# 4. Generate Visualization for Paper
plt.figure(figsize=(10, 6))
plt.hist(curvatures, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(mean_k, color='red', linestyle='dashed', linewidth=2, label=f'Mean κ: {mean_k:.4f}')
plt.title("Intrinsic Curvature Distribution of Mathlib")
plt.xlabel("Estimated Ricci Curvature (κ)")
plt.ylabel("Frequency")
plt.legend()
plt.savefig("manifold_signature.png")
print("\nDistribution plot saved as 'manifold_signature.png'")