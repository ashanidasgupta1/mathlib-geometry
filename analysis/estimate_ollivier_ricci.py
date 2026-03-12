import networkx as nx
import json
import random
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import time

# ── Ollivier-Ricci core ────────────────────────────────────────────────────────

def lazy_distribution(G, node, alpha=0.5):
    """
    Alpha-lazy random walk distribution at `node`.
    Place alpha mass on node itself, spread (1-alpha) uniformly over neighbors.
    """
    neighbors = list(G.neighbors(node))
    deg = len(neighbors)
    dist = {node: alpha}
    if deg > 0:
        for nb in neighbors:
            dist[nb] = dist.get(nb, 0) + (1 - alpha) / deg
    return dist

def earth_movers_distance(mu, nu, dist_matrix, node_index):
    """
    Compute W1(mu, nu) using linear programming (scipy).
    mu, nu: dicts {node: probability}
    dist_matrix: precomputed pairwise distances (numpy array)
    node_index: dict {node: index in dist_matrix}
    """
    from scipy.optimize import linprog

    nodes_mu = list(mu.keys())
    nodes_nu = list(nu.keys())
    n = len(nodes_mu)
    m = len(nodes_nu)

    # Cost matrix
    C = np.zeros((n, m))
    for i, u in enumerate(nodes_mu):
        for j, v in enumerate(nodes_nu):
            ui = node_index.get(u)
            vi = node_index.get(v)
            if ui is not None and vi is not None:
                C[i, j] = dist_matrix[ui, vi]
            else:
                C[i, j] = 0.0

    p = np.array([mu[u] for u in nodes_mu])
    q = np.array([nu[v] for v in nodes_nu])

    # Flatten for linprog
    c = C.flatten()

    # Constraints: row sums = p, col sums = q
    A_eq_rows = np.zeros((n, n * m))
    for i in range(n):
        A_eq_rows[i, i * m:(i + 1) * m] = 1

    A_eq_cols = np.zeros((m, n * m))
    for j in range(m):
        A_eq_cols[j, j::m] = 1

    A_eq = np.vstack([A_eq_rows, A_eq_cols])
    b_eq = np.concatenate([p, q])

    bounds = [(0, None)] * (n * m)

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    return result.fun if result.success else np.nan


def ollivier_ricci_edge(G, u, v, alpha=0.5, dist_matrix=None, node_index=None):
    """Compute Ollivier-Ricci curvature for edge (u, v)."""
    d_uv = dist_matrix[node_index[u], node_index[v]]
    if d_uv == 0:
        return np.nan

    mu_u = lazy_distribution(G, u, alpha)
    mu_v = lazy_distribution(G, v, alpha)

    w1 = earth_movers_distance(mu_u, mu_v, dist_matrix, node_index)
    kappa = 1.0 - w1 / d_uv
    return kappa


# ── Main computation ───────────────────────────────────────────────────────────

def compute_orc(G, label, sample_size=1000, alpha=0.5, max_neighbors=50):
    print(f"\n{'='*55}")
    print(f"Computing ORC for: {label}")
    print(f"{'='*55}")

    all_edges = list(G.edges())
    sampled_edges = random.sample(all_edges, min(sample_size, len(all_edges)))
    print(f"Sampled {len(sampled_edges)} edges")

    # Collect relevant nodes but CAP neighbor lists
    relevant_nodes = set()
    for u, v in sampled_edges:
        relevant_nodes.add(u)
        relevant_nodes.add(v)
        u_neighbors = list(G.neighbors(u))[:max_neighbors]
        v_neighbors = list(G.neighbors(v))[:max_neighbors]
        relevant_nodes.update(u_neighbors)
        relevant_nodes.update(v_neighbors)

    H = G.subgraph(relevant_nodes).copy()
    print(f"Local subgraph: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")
    # ... rest unchanged

    # Precompute all-pairs shortest paths on local subgraph
    print("Computing shortest paths...")
    t0 = time.time()
    node_list = list(H.nodes())
    node_index = {n: i for i, n in enumerate(node_list)}

    # Build sparse adjacency matrix
    row, col, data = [], [], []
    for u2, v2 in H.edges():
        i, j = node_index[u2], node_index[v2]
        row += [i, j]
        col += [j, i]
        data += [1, 1]
    N = len(node_list)
    adj = csr_matrix((data, (row, col)), shape=(N, N))
    dist_matrix = shortest_path(adj, directed=False, unweighted=True)
    print(f"Shortest paths done in {time.time()-t0:.1f}s")

    # Compute ORC for each sampled edge
    print(f"Computing curvature for {len(sampled_edges)} edges...")
    t0 = time.time()
    curvatures = []
    skipped = 0
    for idx, (u, v) in enumerate(sampled_edges):
        if u not in node_index or v not in node_index:
            skipped += 1
            continue
        kappa = ollivier_ricci_edge(G, u, v, alpha=alpha,
                                    dist_matrix=dist_matrix,
                                    node_index=node_index)
        if not np.isnan(kappa):
            curvatures.append(kappa)
        else:
            skipped += 1

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {idx+1}/{len(sampled_edges)} edges done "
                  f"({elapsed:.1f}s, ~{elapsed/(idx+1)*(len(sampled_edges)-idx-1):.0f}s remaining)")

    print(f"\nRESULTS: {label}")
    print(f"  Edges computed:  {len(curvatures)}")
    print(f"  Skipped:         {skipped}")
    print(f"  Mean κ:          {np.mean(curvatures):.4f}")
    print(f"  Median κ:        {np.median(curvatures):.4f}")
    print(f"  Max κ:           {np.max(curvatures):.4f}")
    print(f"  Min κ:           {np.min(curvatures):.4f}")
    print(f"  Std κ:           {np.std(curvatures):.4f}")

    return curvatures


# ── Load graphs ────────────────────────────────────────────────────────────────

print("Loading graph...")
with open("mathlib_graph.json", 'r') as f:
    data = json.load(f)

G_full = nx.Graph()
for entry in data:
    u = entry['name']
    for v in entry['edges']:
        G_full.add_edge(u, v)
print(f"Full graph: {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges")

print("\nBuilding pruned graph...")
G_pruned = G_full.copy()
hubs = [node for node, deg in sorted(G_pruned.degree(),
        key=lambda x: x[1], reverse=True)[:6195]]
G_pruned.remove_nodes_from(hubs)
print(f"Pruned graph: {G_pruned.number_of_nodes()} nodes, {G_pruned.number_of_edges()} edges")

# ── Run ────────────────────────────────────────────────────────────────────────

curvatures_full   = compute_orc(G_full,   "UNPRUNED GRAPH (619,518 nodes)", sample_size=1000)
curvatures_pruned = compute_orc(G_pruned, "PRUNED GRAPH M' (~613,000 nodes)", sample_size=1000)

# ── Comparison ─────────────────────────────────────────────────────────────────

print(f"\n{'='*55}")
print("COMPARISON SUMMARY")
print(f"{'='*55}")
pruned_label = "Pruned M'"
print(f"{'Metric':<12} {'Unpruned':>14} {pruned_label:>14}")
print(f"{'-'*42}")
print(f"{'Mean κ':<12} {np.mean(curvatures_full):>14.4f} {np.mean(curvatures_pruned):>14.4f}")
print(f"{'Median κ':<12} {np.median(curvatures_full):>14.4f} {np.median(curvatures_pruned):>14.4f}")
print(f"{'Max κ':<12} {np.max(curvatures_full):>14.4f} {np.max(curvatures_pruned):>14.4f}")
print(f"{'Min κ':<12} {np.min(curvatures_full):>14.4f} {np.min(curvatures_pruned):>14.4f}")
print(f"{'Std κ':<12} {np.std(curvatures_full):>14.4f} {np.std(curvatures_pruned):>14.4f}")
print(f"{'='*55}")