import torch
import geoopt
import numpy as np

# --- CONFIGURATION ---
NAMESPACE_FILTER = "Topology"  # Change this to "Algebra", "MeasureTheory", etc.
K_NEIGHBORS = 5                # Number of surrounding theorems to define the void

# 1. Load the trained HNN data
print("Loading HNN embeddings...")
checkpoint = torch.load("mathlib_hnn_embeddings.pt")
embeddings = checkpoint['embeddings']
names = checkpoint['names']
ball = geoopt.PoincareBall(c=1.0)

# 2. Filter Nodes by Namespace (The Clique)
print(f"Filtering theorems in the '{NAMESPACE_FILTER}' namespace...")
clique_indices = [i for i, name in enumerate(names) if name.startswith(NAMESPACE_FILTER)]

if not clique_indices:
    print(f"Error: No theorems found starting with '{NAMESPACE_FILTER}'.")
    exit()

clique_indices = torch.tensor(clique_indices, dtype=torch.long)
clique_embeddings = embeddings[clique_indices]

print(f"Found {len(clique_indices)} theorems in the {NAMESPACE_FILTER} clique.")

# 3. Hunt for an Internal Void
# Pick a random "anchor" theorem inside the clique to center our search
start_idx = np.random.choice(len(clique_indices))
anchor_node = clique_embeddings[start_idx]

# Calculate hyperbolic distances from this anchor to all other theorems IN THE CLIQUE
distances = ball.dist(anchor_node, clique_embeddings)

# Find the closest neighbors to define our dense local cluster
dist_values, local_indices = torch.topk(distances, k=K_NEIGHBORS, largest=False)

# 4. Results for LLM Prompting
print("\n" + "="*60)
print(f"INTERNAL CLIQUE VOID FOUND: {NAMESPACE_FILTER.upper()}")
print("="*60)

# The Fréchet mean proxy (using the anchor's coordinates as the center of the void)
print(f"Void Coordinates (16D): \n{anchor_node.numpy()}\n")

print("Surrounding Clique Theorems (Inputs for LLM):")
cluster_names = []
for i in range(K_NEIGHBORS):
    node_idx = clique_indices[local_indices[i]].item()
    name = names[node_idx]
    dist = dist_values[i].item()
    cluster_names.append(name)
    print(f" - {name} (Hyperbolic Dist: {dist:.4f})")

print("\nSuggested LLM Prompt for Refactoring:")
print(f"\"I am analyzing the Lean 4 Mathlib library. The following theorems form a dense geometric cluster in the '{NAMESPACE_FILTER}' namespace: {', '.join(cluster_names)}. They are extremely close in logical space. Can you propose a general 'helper lemma' or abstract definition that sits directly between them and could be used to shorten all of their proofs?\"")