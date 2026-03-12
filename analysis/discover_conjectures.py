import torch
import geoopt
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# 1. Load the trained HNN data
print("Loading HNN embeddings...")
checkpoint = torch.load("mathlib_hnn_embeddings.pt")
embeddings = checkpoint['embeddings']
names = checkpoint['names']
ball = geoopt.PoincareBall(c=1.0)

# 2. Identify Peripheral Nodes
# In the Poincare ball, radius (norm) represents how specialized/peripheral a node is.
norms = torch.norm(embeddings, dim=1)
# We look for nodes in the outer shell (radius > 0.8)
peripheral_mask = norms > 0.8
peripheral_indices = torch.where(peripheral_mask)[0]
peripheral_embeddings = embeddings[peripheral_indices]

print(f"Found {len(peripheral_indices)} peripheral theorems near the boundary.")

# 3. Find Clusters in the "Hyperbolic Sky"
# We pick a random starting "star" from the periphery
start_idx = np.random.choice(len(peripheral_indices))
target_node = peripheral_embeddings[start_idx]

# Calculate hyperbolic distances from this star to all other peripheral stars
# Formula: d(u, v) = ball.dist(u, v)
distances = ball.dist(target_node, peripheral_embeddings)

# Find the 5 nearest neighbors in the hyperbolic space
nearest_k = 5
dist_values, local_indices = torch.topk(distances, k=nearest_k, largest=False)

# 4. Results for LLM Prompting
print("\n" + "="*50)
print("HYPERBOLIC CONJECTURE CLUSTER FOUND")
print("="*50)
print(f"Centroid 'Void' Coordinates (16D): \n{target_node.numpy()}\n")

print("Peripheral Stars (Inputs for LLM):")
cluster_names = []
for i in range(nearest_k):
    node_idx = peripheral_indices[local_indices[i]]
    name = names[node_idx]
    dist = dist_values[i].item()
    cluster_names.append(name)
    print(f" - {name} (Hyperbolic Dist: {dist:.4f})")

print("\nSuggested LLM Prompt:")
print(f"\"Synthesize a unifying conjecture for the following Lean Mathlib theorems: {', '.join(cluster_names)}. These theorems are proximal in a 16D hyperbolic manifold, suggesting a latent logical bridge.\"")