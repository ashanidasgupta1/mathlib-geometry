import json
import networkx as nx
import numpy as np
import random
from tqdm import tqdm

def load_graph(json_path):
    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    G = nx.DiGraph()
    
    print("Building NetworkX graph...")
    # Mathlib is large, so this might take 10-20 seconds
    for entry in tqdm(data):
        u = entry['name']
        G.add_node(u, type=entry['type'])
        for v in entry['edges']:
            # Only add edge if target node is also in the graph (clean data)
            G.add_edge(u, v)
            
    print(f"Nodes: {len(G)}")
    print(f"Edges: {len(G.edges())}")

    # Convert to undirected for Gromov calculation
    # (Distance in hyperbolic space is usually symmetric)
    print("Extracting largest connected component...")
    G_undir = G.to_undirected()
    components = sorted(nx.connected_components(G_undir), key=len, reverse=True)
    G_main = G_undir.subgraph(components[0]).copy()
    print(f"Main Component Nodes: {len(G_main)}")
    
    return G_main

def estimate_delta(G, num_samples=1000):
    """
    Estimates Gromov delta using the 4-point condition.
    For a graph to be hyperbolic, delta should be small relative to diameter.
    """
    print(f"\nSampling {num_samples} quadruplets for Gromov Delta...")
    nodes = list(G.nodes())
    deltas = []
    
    # We sample 4 random nodes (x, y, z, w) and check the "slim triangle" condition
    for _ in tqdm(range(num_samples)):
        try:
            q = random.sample(nodes, 4)
            # Calculate all pairwise distances
            d = {}
            # We only need 3 sums: S1=xy+zw, S2=xz+yw, S3=xw+yz
            # S = Sorted(S1, S2, S3)
            # Delta = (Largest - Middle) / 2
            
            # Helper to get dist
            def get_dist(n1, n2):
                return nx.shortest_path_length(G, n1, n2)

            s1 = get_dist(q[0], q[1]) + get_dist(q[2], q[3]) # xy + zw
            s2 = get_dist(q[0], q[2]) + get_dist(q[1], q[3]) # xz + yw
            s3 = get_dist(q[0], q[3]) + get_dist(q[1], q[2]) # xw + yz
            
            sums = sorted([s1, s2, s3])
            delta = (sums[2] - sums[1]) / 2.0
            deltas.append(delta)
            
        except (nx.NetworkXNoPath, IndexError):
            continue

    return np.mean(deltas), np.max(deltas)

if __name__ == "__main__":
    # 1. Load Data
    G = load_graph("mathlib_graph.json")
    
    # 2. Estimate Hyperbolicity
    mean_delta, max_delta = estimate_delta(G, num_samples=2000)
    
    print(f"\n--- RESULTS ---")
    print(f"Mean Gromov Delta: {mean_delta:.4f}")
    print(f"Max Delta: {max_delta:.4f}")
    
    if mean_delta < 2.0:
        print("CONCLUSION: Extremely Hyperbolic (Tree-like). Ideal for HNN.")
    elif mean_delta < 5.0:
        print("CONCLUSION: Moderately Hyperbolic. HNN should outperform Euclidean GNN.")
    else:
        print("CONCLUSION: Low Hyperbolicity (Grid-like). HNN might not help.")