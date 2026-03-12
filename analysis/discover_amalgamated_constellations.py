"""
discover_amalgamated_constellations.py  (v3 — TF-IDF weighted C)

Fixes from v2:
  - Shared deps weighted by inverse document frequency (IDF)
  - High-degree foundational nodes (Add, AddCommGroup etc) filtered out
  - Score rewards SPECIFIC shared deps, not generic ones
  - Same Complex constellation can't dominate all matches
"""

import torch
import geoopt
import numpy as np
import json
import math
from collections import defaultdict, Counter

# ── CONFIG ─────────────────────────────────────────────────────
EMBEDDING_FILE = "mathlib_hnn_embeddings.pt"
GRAPH_FILE = "mathlib_graph.json"
K = 5
NUM_CLUSTERS = 15
ANCHORS_PER_SEARCH = 30
TOP_MATCHES = 10
SHAPE_TOLERANCE = 0.05
MIN_WEIGHTED_SCORE = 5.0        # minimum IDF-weighted C score
MAX_DIRECT_EDGES = 1
ANCESTOR_DEPTH = 2
MAX_DEP_FREQUENCY = 0.01        # deps used by >1% of all nodes = noise
TOP_C_TO_SHOW = 20              # most specific shared deps to display

DOMAINS = [
    ("MeasureTheory",       "Measure theory"),
    ("ProbabilityTheory",   "Probability theory"),
    ("Polynomial",          "Polynomials"),
    ("LinearMap",           "Linear maps"),
    ("Matrix",              "Matrices"),
    ("Subgroup",            "Group theory"),
    ("Ideal",               "Ideal theory"),
    ("Module",              "Modules"),
    ("Matroid",             "Matroids"),
    ("SimpleGraph",         "Graph theory"),
    ("Filter",              "Filters"),
    ("TopologicalSpace",    "Topological spaces"),
    ("Complex",             "Complex analysis"),
    ("Finset",              "Finite sets"),
    ("NumberField",         "Number fields"),
    ("ContinuousLinearMap", "Continuous linear maps"),
    ("AlgebraicGeometry",   "Algebraic geometry"),
    ("CategoryTheory",      "Category theory"),
]

# ── LOAD EMBEDDINGS ────────────────────────────────────────────
print("Loading embeddings...")
checkpoint = torch.load(EMBEDDING_FILE, weights_only=True)
embeddings = checkpoint['embeddings']
names = checkpoint['names']
ball = geoopt.PoincareBall(c=1.0)
norms = torch.norm(embeddings, dim=1)
norm_threshold = torch.quantile(norms, 0.3).item()
N_TOTAL = len(names)
print(f"  {N_TOTAL} nodes, dim={embeddings.shape[1]}")

# ── LOAD GRAPH + COMPUTE IDF ───────────────────────────────────
print("Loading dependency graph...")
with open(GRAPH_FILE, 'r') as f:
    graph_data = json.load(f)

deps_of = defaultdict(set)
used_by_count = Counter()

for entry in graph_data:
    u = entry['name']
    for v in entry.get('edges', entry.get('deps', [])):
        deps_of[u].add(v)
        used_by_count[v] += 1

print(f"  {len(deps_of)} nodes with dependencies")

# Compute IDF for every node
# IDF(node) = log(N / (1 + count(node)))
# High IDF = rare/specific, Low IDF = common/foundational
node_idf = {}
for n in names:
    freq = used_by_count.get(n, 0)
    node_idf[n] = math.log(N_TOTAL / (1 + freq))

# Frequency threshold: skip nodes used by > MAX_DEP_FREQUENCY of all
freq_threshold = int(N_TOTAL * MAX_DEP_FREQUENCY)
print(f"  Frequency threshold: {freq_threshold} "
      f"(deps used by >{freq_threshold} nodes are noise)")

# Show some examples
print("  Examples of high-IDF (specific) nodes:")
specific = sorted(node_idf.items(), key=lambda x: -x[1])
for n, idf in specific[:5]:
    print(f"    {n}: IDF={idf:.2f}, used_by={used_by_count.get(n, 0)}")
print("  Examples of low-IDF (foundational) nodes:")
generic = sorted(node_idf.items(), key=lambda x: x[1])
for n, idf in generic[:5]:
    print(f"    {n}: IDF={idf:.2f}, used_by={used_by_count.get(n, 0)}")


def get_ancestors(node, depth=ANCESTOR_DEPTH):
    visited = set()
    frontier = {node}
    for _ in range(depth):
        next_frontier = set()
        for n in frontier:
            for dep in deps_of.get(n, set()):
                if dep not in visited:
                    visited.add(dep)
                    next_frontier.add(dep)
        frontier = next_frontier
    return visited


def get_cluster_ancestors(cluster_names):
    all_ancestors = set()
    for n in cluster_names:
        all_ancestors |= get_ancestors(n)
    all_ancestors -= set(cluster_names)
    return all_ancestors


def filter_and_weight_shared(shared, prefix_a, prefix_b):
    """
    Filter out high-frequency noise, weight remaining by IDF.
    Return list of (name, idf_score) sorted by specificity.
    """
    weighted = []
    for dep in shared:
        freq = used_by_count.get(dep, 0)
        # Skip foundational noise
        if freq > freq_threshold:
            continue
        # Skip Lean/Std internals
        if dep.startswith(("Lean.", "Std.", "Aesop.", "Batteries.")):
            continue
        idf = node_idf.get(dep, 0)
        weighted.append((dep, idf, freq))

    weighted.sort(key=lambda x: -x[1])  # most specific first
    return weighted


def count_direct_edges(names_a, names_b):
    set_a, set_b = set(names_a), set(names_b)
    count = 0
    for a in set_a:
        count += len(deps_of.get(a, set()) & set_b)
    for b in set_b:
        count += len(deps_of.get(b, set()) & set_a)
    return count


# ── GEOMETRY ───────────────────────────────────────────────────

def get_peripheral_indices(prefix):
    return [i for i, n in enumerate(names)
            if n.startswith(prefix) and norms[i].item() >= norm_threshold]


def pairwise_dist_matrix(indices):
    embs = embeddings[torch.tensor(indices, dtype=torch.long)]
    n = len(indices)
    D = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            d = ball.dist(embs[i].unsqueeze(0), embs[j].unsqueeze(0)).item()
            D[i, j] = d
            D[j, i] = d
    return D


def upper_triangle(D):
    n = D.shape[0]
    return sorted(D[i, j].item() for i in range(n) for j in range(i + 1, n))


def shape_distance(sig_a, sig_b):
    a, b = np.array(sig_a), np.array(sig_b)
    mean_val = (a.mean() + b.mean()) / 2
    if mean_val < 0.01:
        return float('inf')
    return np.mean(np.abs(a - b)) / mean_val


# ── PHASE 1: Find clusters ────────────────────────────────────
print("\nPhase 1: Finding clusters...")
all_clusters = {}

for prefix, domain_name in DOMAINS:
    indices = get_peripheral_indices(prefix)
    if len(indices) < K:
        print(f"  {domain_name}: SKIPPED")
        continue

    idx_tensor = torch.tensor(indices, dtype=torch.long)
    domain_embs = embeddings[idx_tensor]
    used = set()
    clusters = []

    for _ in range(NUM_CLUSTERS):
        available = [i for i in range(len(indices)) if i not in used]
        if len(available) < K:
            break

        best = None
        best_mean = float('inf')

        for _ in range(ANCHORS_PER_SEARCH):
            start = np.random.choice(available)
            anchor = domain_embs[start]
            avail_t = torch.tensor(available, dtype=torch.long)
            avail_embs = domain_embs[avail_t]
            dists = ball.dist(anchor, avail_embs)
            vals, local_idx = torch.topk(dists, k=K, largest=False)
            mean_d = vals.mean().item()
            if mean_d < best_mean:
                best_mean = mean_d
                best = (vals, local_idx, avail_t)

        if best is None:
            break

        vals, local_idx, avail_t = best
        cluster_indices = []
        cluster_names_list = []

        for i in range(K):
            avail_i = local_idx[i].item()
            global_i = indices[avail_t[avail_i].item()]
            cluster_indices.append(global_i)
            cluster_names_list.append(names[global_i])
            used.add(avail_t[avail_i].item())

        D = pairwise_dist_matrix(cluster_indices)
        sig = upper_triangle(D)
        ancestors = get_cluster_ancestors(cluster_names_list)

        clusters.append({
            "domain": prefix,
            "domain_name": domain_name,
            "indices": cluster_indices,
            "names": cluster_names_list,
            "mean_dist": best_mean,
            "signature": sig,
            "ancestors": ancestors,
        })

    if clusters:
        all_clusters[prefix] = clusters
        print(f"  {domain_name}: {len(clusters)} clusters")

# ── PHASE 2: Match with shape + IDF-weighted C ────────────────
print("\nPhase 2: Matching (shape + IDF-weighted shared deps)...")
matches = []
seen_pairs = set()  # prevent same constellation dominating

domain_keys = list(all_clusters.keys())
for i, key_a in enumerate(domain_keys):
    for key_b in domain_keys[i + 1:]:
        for ca in all_clusters[key_a]:
            for cb in all_clusters[key_b]:
                # Check shape
                sd = shape_distance(ca["signature"], cb["signature"])
                if sd > SHAPE_TOLERANCE:
                    continue

                # Check direct edges
                direct = count_direct_edges(ca["names"], cb["names"])
                if direct > MAX_DIRECT_EDGES:
                    continue

                # Compute IDF-weighted C
                shared_raw = ca["ancestors"] & cb["ancestors"]
                weighted_c = filter_and_weight_shared(
                    shared_raw, key_a, key_b
                )

                # Sum of IDF scores = quality of C
                idf_score = sum(idf for _, idf, _ in weighted_c)
                if idf_score < MIN_WEIGHTED_SCORE:
                    continue

                # Prevent same constellation appearing too many times
                key_sig = (
                    frozenset(ca["names"]),
                    frozenset(cb["names"])
                )
                if key_sig in seen_pairs:
                    continue
                seen_pairs.add(key_sig)

                matches.append({
                    "cluster_a": ca,
                    "cluster_b": cb,
                    "shape_distance": sd,
                    "weighted_c": weighted_c,
                    "idf_score": idf_score,
                    "n_specific_deps": len(weighted_c),
                    "direct_edges": direct,
                })

matches.sort(key=lambda x: x["idf_score"], reverse=True)
print(f"  Found {len(matches)} valid matches")

# ── PHASE 3: Report ───────────────────────────────────────────
print(f"\n{'='*60}")
print("AMALGAMATED CONSTELLATIONS (TF-IDF weighted)")
print(f"{'='*60}")

if not matches:
    print("  No matches found.")
    print("  Try: raise SHAPE_TOLERANCE, lower MIN_WEIGHTED_SCORE,")
    print("       or raise MAX_DEP_FREQUENCY")
else:
    for rank, m in enumerate(matches[:TOP_MATCHES], 1):
        ca = m["cluster_a"]
        cb = m["cluster_b"]
        wc = m["weighted_c"]

        print(f"\n{'='*60}")
        print(f"MATCH #{rank}  (IDF_score={m['idf_score']:.1f}, "
              f"shape={m['shape_distance']:.4f}, "
              f"|C_specific|={m['n_specific_deps']}, "
              f"direct={m['direct_edges']})")
        print(f"  {ca['domain_name']} (mean={ca['mean_dist']:.3f})")
        print(f"  {cb['domain_name']} (mean={cb['mean_dist']:.3f})")
        print(f"{'='*60}")

        print(f"\n  Constellation A ({ca['domain']}):")
        for n in ca["names"]:
            print(f"    - {n}")

        print(f"\n  Constellation B ({cb['domain']}):")
        for n in cb["names"]:
            print(f"    - {n}")

        # Show top specific shared deps
        top_c = wc[:TOP_C_TO_SHOW]
        print(f"\n  Most specific shared deps (C):")
        for dep_name, idf, freq in top_c:
            print(f"    * {dep_name}  "
                  f"(IDF={idf:.2f}, used_by={freq})")

        # ── LLM Prompt ────────────────────────────────────────
        c_list = "\n".join(
            f"  * {dep_name} (specificity: {idf:.1f}, "
            f"used by {freq} declarations)"
            for dep_name, idf, freq in top_c
        )
        prompt = f"""I am analyzing the Lean 4 Mathlib library.

I found two constellations in DIFFERENT domains with nearly
identical geometric structure AND specific shared dependencies
(not just foundational typeclasses — these are meaningful
mathematical connections).

Constellation A ({ca['domain_name']}):
"""
        for n in ca["names"]:
            prompt += f"  - {n}\n"
        prompt += f"""
Constellation B ({cb['domain_name']}):
"""
        for n in cb["names"]:
            prompt += f"  - {n}\n"
        prompt += f"""
Most specific shared dependencies (the "C" mediating both):
{c_list}

Shape distance: {m['shape_distance']:.4f}
Direct edges between A and B: {m['direct_edges']}
Specific shared deps: {m['n_specific_deps']}

These shared dependencies are NOT generic foundations like
AddCommGroup — they are specific mathematical lemmas that both
constellations build on independently.

Please propose:
1. What the specific shared deps reveal about the mathematical
   relationship between these two domains
2. A NEW theorem in the "amalgamation" that uses concepts from
   BOTH domains, mediated through the shared structure
3. A sketch in Lean 4 syntax
"""
        print(f"\n  [Prompt generated, see amalgamated_prompts.txt]")
        m["prompt"] = prompt

    # Save
    with open("amalgamated_prompts.txt", "w") as f:
        for rank, m in enumerate(matches[:TOP_MATCHES], 1):
            f.write(f"{'='*60}\n")
            f.write(f"MATCH #{rank} (IDF={m['idf_score']:.1f})\n")
            f.write(f"{'='*60}\n\n")
            f.write(m["prompt"])
            f.write("\n\n")
    print(f"\nPrompts saved to amalgamated_prompts.txt")

print(f"\nDone. {len(matches)} matches found.")