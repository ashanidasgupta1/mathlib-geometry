import networkx as nx
import json
import random
import numpy as np

# 1. Load the Pruned Graph (or Global Graph)
print("Loading graph for Gromov analysis...")
# Adjust filename as needed (e.g., mathlib_graph.json)
with open("mathlib_graph.json", 'r') as f:
    data = json.load(f)

G = nx.Graph()
for entry in data:
    u = entry['name']
    for v in entry['edges']:
        G.add_edge(u, v)
        
# hubs = [node for node, deg in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:6195]]
# G.remove_nodes_from(hubs)

nodes = list(G.nodes())
print(f"Graph ready: {len(nodes)} nodes.")

def get_delta_sample(sample_size=1000):
    deltas = []
    skipped = 0
    print(f"Sampling {sample_size} quadruples...")
    
    for i in range(sample_size):
        # Pick 4 random nodes
        quad = random.sample(nodes, 4)
        a, b, c, d = quad
        
        try:
            # We need 6 distances to check the 4-point condition
            dists = {
                'ab': nx.shortest_path_length(G, a, b),
                'cd': nx.shortest_path_length(G, c, d),
                'ac': nx.shortest_path_length(G, a, c),
                'bd': nx.shortest_path_length(G, b, d),
                'ad': nx.shortest_path_length(G, a, d),
                'bc': nx.shortest_path_length(G, b, c)
            }

            # ADD THIS:
            if i < 5:
                print(f"  Sample distances for quadruple {i}: {dists}")
            
            sums = sorted([
                dists['ab'] + dists['cd'],
                dists['ac'] + dists['bd'],
                dists['ad'] + dists['bc']
            ], reverse=True)
            
            # 4-point condition: (Max Sum - Mid Sum) / 2
            delta = (sums[0] - sums[1]) / 2.0
            deltas.append(delta)
            
        except nx.NetworkXNoPath:
            skipped += 1
            continue
            
        if i % 500 == 0:
            print(f"  Processed {i}/{sample_size}...")

    print(f"\nValid quadruples computed: {len(deltas)}")
    print(f"Skipped (disconnected):    {skipped}")
    return deltas

# Execute
sample_results = get_delta_sample(10000)
max_delta = max(sample_results)
avg_delta = np.mean(sample_results)

print("\n" + "="*45)
print("GROMOV HYPERBOLICITY ESTIMATE (STOCHASTIC)")
print("="*45)
print(f"Max sampled δ: {max_delta:.2f}")
print(f"Mean sampled δ: {avg_delta:.4f}")
print(f"Relative δ (δ/Diameter): {max_delta / nx.diameter(G) if len(nodes) < 5000 else 'Calc Diameter too slow'}")
print("="*45)