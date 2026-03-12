"""
null_model_tests.py

Generate three null model graphs matching Mathlib's basic statistics,
apply the same 1% pruning, and measure delta and kappa on each.

This answers: is the Tree of Cliques special to mathematics, or
would any large graph with similar degree statistics look the same?

"""
QUICK_TEST = True
import networkx as nx
import numpy as np
import random
import json
import time

# ── CONFIG ─────────────────────────────────────────────────────
GRAPH_FILE = "mathlib_graph.json"
DELTA_SAMPLES = 1000        # quadruples for delta estimation
KAPPA_SAMPLES = 5000        # edges for kappa estimation
PRUNE_PERCENTILE = 0.99     # remove top 1% by degree
QUICK_TEST = False          # set True to use smaller graphs first

# If QUICK_TEST, use smaller graphs for fast iteration
QUICK_NODES = 10000
QUICK_EDGES_PER_NODE = 21

# ── LOAD MATHLIB GRAPH ─────────────────────────────────────────
print("=" * 60)
print("LOADING MATHLIB GRAPH")
print("=" * 60)

print("Loading graph...")
with open(GRAPH_FILE, 'r') as f:
    data = json.load(f)

G_mathlib = nx.Graph()
for entry in data:
    u = entry['name']
    for v in entry.get('edges', entry.get('deps', [])):
        G_mathlib.add_edge(u, v)

n_nodes_mathlib = G_mathlib.number_of_nodes()
n_edges_mathlib = G_mathlib.number_of_edges()
degrees_mathlib = sorted([d for _, d in G_mathlib.degree()], reverse=True)
mean_degree = np.mean(degrees_mathlib)
max_degree = degrees_mathlib[0]

print(f"  Nodes: {n_nodes_mathlib}")
print(f"  Edges: {n_edges_mathlib}")
print(f"  Mean degree: {mean_degree:.1f}")
print(f"  Max degree: {max_degree}")
print(f"  Median degree: {np.median(degrees_mathlib):.0f}")


# ── MEASUREMENT FUNCTIONS ──────────────────────────────────────

def prune_top_percent(G, percentile=PRUNE_PERCENTILE):
    """Remove top (1-percentile)*100% of nodes by degree."""
    degrees = dict(G.degree())
    threshold = np.quantile(list(degrees.values()), percentile)
    to_remove = [n for n, d in degrees.items() if d > threshold]
    G_pruned = G.copy()
    G_pruned.remove_nodes_from(to_remove)
    print(f"    Pruned {len(to_remove)} nodes "
          f"(threshold degree > {threshold:.0f})")
    print(f"    Remaining: {G_pruned.number_of_nodes()} nodes, "
          f"{G_pruned.number_of_edges()} edges")
    return G_pruned


def estimate_delta(G, n_samples=DELTA_SAMPLES):
    """
    Estimate Gromov hyperbolicity via random quadruple sampling.
    Same method as check_hyperbolicity.py / estimate_gromov_delta.py.
    """
    # Work on largest connected component
    if not nx.is_connected(G):
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        G = G.subgraph(components[0]).copy()
        print(f"    Using largest component: {G.number_of_nodes()} nodes")

    nodes = list(G.nodes())
    if len(nodes) < 4:
        return float('nan')

    deltas = []
    failed = 0

    for _ in range(n_samples):
        quad = random.sample(nodes, 4)
        a, b, c, d = quad
        try:
            dists = {
                'ab': nx.shortest_path_length(G, a, b),
                'cd': nx.shortest_path_length(G, c, d),
                'ac': nx.shortest_path_length(G, a, c),
                'bd': nx.shortest_path_length(G, b, d),
                'ad': nx.shortest_path_length(G, a, d),
                'bc': nx.shortest_path_length(G, b, c),
            }
            sums = sorted([
                dists['ab'] + dists['cd'],
                dists['ac'] + dists['bd'],
                dists['ad'] + dists['bc']
            ], reverse=True)
            delta = (sums[0] - sums[1]) / 2.0
            deltas.append(delta)
        except nx.NetworkXNoPath:
            failed += 1
            continue

    if not deltas:
        return float('nan')

    max_delta = max(deltas)
    mean_delta = np.mean(deltas)
    print(f"    Delta: max={max_delta:.2f}, mean={mean_delta:.4f} "
          f"({len(deltas)} samples, {failed} failed)")
    return max_delta


def estimate_kappa(G, n_samples=KAPPA_SAMPLES):
    """
    Estimate Ollivier-Ricci curvature via Jaccard overlap proxy.
    Same method as estimate_ollivier_ricci.py / estimate_curvature.py.
    """
    edges = list(G.edges())
    if len(edges) < n_samples:
        sample_edges = edges
    else:
        sample_edges = random.sample(edges, n_samples)

    kappas = []
    for u, v in sample_edges:
        set_u = set(G.neighbors(u))
        set_v = set(G.neighbors(v))
        intersection = len(set_u & set_v)
        union = len(set_u | set_v)
        if union == 0:
            continue
        kappa = intersection / union - 1 / (len(set_u) + len(set_v))
        kappas.append(kappa)

    if not kappas:
        return float('nan'), float('nan')

    mean_kappa = np.mean(kappas)
    std_kappa = np.std(kappas)
    print(f"    Kappa: mean={mean_kappa:.4f}, std={std_kappa:.4f} "
          f"({len(kappas)} samples)")
    return mean_kappa, std_kappa


def measure_graph(G, name):
    """Run all measurements on a graph."""
    print(f"\n  Measuring {name}...")
    print(f"    Nodes: {G.number_of_nodes()}, "
          f"Edges: {G.number_of_edges()}")

    # Prune
    print(f"  Pruning {name}...")
    G_pruned = prune_top_percent(G)

    # Measure delta
    print(f"  Estimating delta for {name} (pruned)...")
    t0 = time.time()
    delta = estimate_delta(G_pruned)
    t_delta = time.time() - t0
    print(f"    Time: {t_delta:.1f}s")

    # Measure kappa
    print(f"  Estimating kappa for {name} (pruned)...")
    t0 = time.time()
    mean_kappa, std_kappa = estimate_kappa(G_pruned)
    t_kappa = time.time() - t0
    print(f"    Time: {t_kappa:.1f}s")

    return {
        "name": name,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "nodes_pruned": G_pruned.number_of_nodes(),
        "edges_pruned": G_pruned.number_of_edges(),
        "delta": delta,
        "mean_kappa": mean_kappa,
        "std_kappa": std_kappa,
    }


# ── MEASURE MATHLIB ────────────────────────────────────────────
print("\n" + "=" * 60)
print("MEASURING MATHLIB (control)")
print("=" * 60)
results = []
results.append(measure_graph(G_mathlib, "Mathlib"))


# ── NULL MODEL 1: ERDOS-RENYI ─────────────────────────────────
print("\n" + "=" * 60)
print("NULL MODEL 1: ERDOS-RENYI")
print("=" * 60)

if QUICK_TEST:
    n_er = QUICK_NODES
    m_er = QUICK_NODES * int(mean_degree / 2)
else:
    n_er = n_nodes_mathlib
    m_er = n_edges_mathlib

print(f"Generating G(n={n_er}, m={m_er})...")
t0 = time.time()
G_er = nx.gnm_random_graph(n_er, m_er)
print(f"  Generated in {time.time() - t0:.1f}s")
results.append(measure_graph(G_er, "Erdos-Renyi"))
del G_er  # free memory


# ── NULL MODEL 2: BARABASI-ALBERT ─────────────────────────────
print("\n" + "=" * 60)
print("NULL MODEL 2: BARABASI-ALBERT")
print("=" * 60)

# m parameter: each new node adds m edges
# Total edges ≈ m * n, so m ≈ n_edges / n_nodes
if QUICK_TEST:
    n_ba = QUICK_NODES
    m_ba = QUICK_EDGES_PER_NODE
else:
    n_ba = n_nodes_mathlib
    m_ba = max(1, int(n_edges_mathlib / n_nodes_mathlib))

print(f"Generating BA(n={n_ba}, m={m_ba})...")
t0 = time.time()
G_ba = nx.barabasi_albert_graph(n_ba, m_ba)
print(f"  Generated in {time.time() - t0:.1f}s")
results.append(measure_graph(G_ba, "Barabasi-Albert"))
del G_ba


# ── NULL MODEL 3: CONFIGURATION MODEL ─────────────────────────
print("\n" + "=" * 60)
print("NULL MODEL 3: CONFIGURATION MODEL")
print("=" * 60)
print("(Same degree sequence as Mathlib, random wiring)")

if QUICK_TEST:
    # Use a truncated degree sequence
    deg_seq = degrees_mathlib[:QUICK_NODES]
    # Ensure even sum
    if sum(deg_seq) % 2 != 0:
        deg_seq[-1] += 1
else:
    deg_seq = [d for _, d in G_mathlib.degree()]
    if sum(deg_seq) % 2 != 0:
        deg_seq[-1] += 1

print(f"Generating configuration model "
      f"(n={len(deg_seq)}, sum_deg={sum(deg_seq)})...")
print("  (This may take a while and use significant RAM...)")
t0 = time.time()
try:
    G_config = nx.configuration_model(deg_seq)
    # Remove self-loops and parallel edges
    G_config = nx.Graph(G_config)
    G_config.remove_edges_from(nx.selfloop_edges(G_config))
    print(f"  Generated in {time.time() - t0:.1f}s")
    results.append(measure_graph(G_config, "Configuration"))
    del G_config
except MemoryError:
    print("  OUT OF MEMORY. Skipping configuration model.")
    print("  Try setting QUICK_TEST = True or reducing graph size.")
    results.append({
        "name": "Configuration",
        "nodes": len(deg_seq),
        "edges": "OOM",
        "delta": "OOM",
        "mean_kappa": "OOM",
        "std_kappa": "OOM",
    })


# ── RESULTS TABLE ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

header = f"{'Model':<22} {'Nodes':>10} {'Edges':>12} " \
         f"{'delta':>8} {'kappa':>8} {'std_k':>8}"
print(header)
print("-" * len(header))

for r in results:
    delta_str = f"{r['delta']:.2f}" if isinstance(r['delta'], float) \
        else str(r['delta'])
    kappa_str = f"{r['mean_kappa']:.4f}" \
        if isinstance(r.get('mean_kappa'), float) \
        else str(r.get('mean_kappa', '?'))
    std_str = f"{r.get('std_kappa', 0):.4f}" \
        if isinstance(r.get('std_kappa'), float) \
        else str(r.get('std_kappa', '?'))

    nodes_str = f"{r.get('nodes_pruned', r['nodes'])}" \
        if 'nodes_pruned' in r else str(r['nodes'])
    edges_str = f"{r.get('edges_pruned', r['edges'])}" \
        if 'edges_pruned' in r else str(r['edges'])

    print(f"{r['name']:<22} {nodes_str:>10} {edges_str:>12} "
          f"{delta_str:>8} {kappa_str:>8} {std_str:>8}")

# ── INTERPRETATION ─────────────────────────────────────────────
print(f"\n{'='*60}")
print("INTERPRETATION")
print("=" * 60)

mathlib_r = results[0]
for r in results[1:]:
    if not isinstance(r.get('mean_kappa'), float):
        continue
    name = r['name']
    dk = r['mean_kappa'] - mathlib_r['mean_kappa']
    dd = r['delta'] - mathlib_r['delta'] if isinstance(r['delta'], float) \
        else float('nan')
    print(f"\n  {name} vs Mathlib:")
    print(f"    delta difference: {dd:+.2f} "
          f"({'higher' if dd > 0 else 'lower'})")
    print(f"    kappa difference: {dk:+.4f} "
          f"({'higher' if dk > 0 else 'lower'})")

    if abs(dk) < 0.005:
        print(f"    -> kappa is SIMILAR: Tree of Cliques may be "
              f"a generic graph property")
    else:
        print(f"    -> kappa is DIFFERENT: Tree of Cliques appears "
              f"specific to Mathlib")

# Save results
with open("null_model_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to null_model_results.json")

print(f"\nDone. Tested {len(results)} models total.")
