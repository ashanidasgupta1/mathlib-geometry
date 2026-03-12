import torch
import geoopt
import numpy as np

EMBEDDING_FILE = "mathlib_hnn_embeddings.pt"
NAMESPACE_FILTER = "Algebra"
K_NEIGHBORS = 5
NUM_CONSTELLATIONS = 5
ANCHORS_PER_SEARCH = 20

print("Loading embeddings...")
checkpoint = torch.load(EMBEDDING_FILE, weights_only=True)
embeddings = checkpoint['embeddings']
names = checkpoint['names']
ball = geoopt.PoincareBall(c=1.0)

clique_indices = [i for i, name in enumerate(names) if name.startswith(NAMESPACE_FILTER)]
clique_indices = torch.tensor(clique_indices, dtype=torch.long)
clique_embeddings = embeddings[clique_indices]
print(f"Found {len(clique_indices)} nodes in '{NAMESPACE_FILTER}'.")

used_indices = set()
constellations = []

for c_num in range(NUM_CONSTELLATIONS):
    available = [i for i in range(len(clique_indices)) if i not in used_indices]
    if len(available) < K_NEIGHBORS:
        print(f"Not enough unused nodes for constellation {c_num+1}. Stopping.")
        break

    best = None
    best_mean_dist = float('inf')

    for _ in range(ANCHORS_PER_SEARCH):
        start_idx = np.random.choice(available)
        anchor = clique_embeddings[start_idx]
        avail_tensor = torch.tensor(available, dtype=torch.long)
        avail_embeddings = clique_embeddings[avail_tensor]
        distances = ball.dist(anchor, avail_embeddings)
        dist_values, local_indices = torch.topk(distances, k=K_NEIGHBORS, largest=False)
        mean_dist = dist_values.mean().item()

        if mean_dist < best_mean_dist:
            best_mean_dist = mean_dist
            best = (start_idx, dist_values, local_indices, avail_tensor)

    start_idx, dist_values, local_indices, avail_tensor = best
    cluster = []
    for i in range(K_NEIGHBORS):
        avail_idx = local_indices[i].item()
        node_idx = clique_indices[avail_tensor[avail_idx]].item()
        name = names[node_idx]
        dist = dist_values[i].item()
        cluster.append((name, dist))
        used_indices.add(avail_tensor[avail_idx].item())

    constellations.append((best_mean_dist, cluster))

    print(f"\n{'='*60}")
    print(f"CONSTELLATION {c_num+1} (mean dist: {best_mean_dist:.4f})")
    print('='*60)
    for name, dist in cluster:
        print(f"  - {name} (dist: {dist:.4f})")

    cluster_names = [n for n, d in cluster]
    print(f"\nLLM PROMPT:")
    print(f"I am analyzing the Lean 4 Mathlib library.")
    print(f"The following theorems form a tight geometric cluster in")
    print(f"the '{NAMESPACE_FILTER}' namespace in hyperbolic logical space:\n")
    for name in cluster_names:
        print(f"  - {name}")
    print(f"\nPropose a helper lemma or abstraction that unifies these.")

print(f"\n\nSUMMARY: Found {len(constellations)} constellations in '{NAMESPACE_FILTER}'")
