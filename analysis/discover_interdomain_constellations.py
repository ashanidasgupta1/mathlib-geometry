"""
(v3)

Find constellations that straddle TWO mathematical domains by
searching for voids along hyperbolic geodesics between them.

Key fixes over v2:
  - Constellation members are drawn ONLY from domain A ∪ B
    (no more Decidable.peirce or Int.shiftRight in results)
  - Peripherality filter: skip nodes near the ball's center
  - Require ≥2 members from each domain
"""

import torch
import geoopt
import numpy as np

# ── CONFIG ─────────────────────────────────────────────────────
EMBEDDING_FILE = "mathlib_hnn_embeddings.pt"
K_NEIGHBORS = 5
MIN_FROM_EACH = 2               # at least 2 from each domain
NUM_CONSTELLATIONS = 3           # per domain pair
PAIRS_TO_SAMPLE = 500            # cross-domain pairs to try
GEODESIC_POINTS = 7              # points sampled per geodesic
MIN_PAIR_DIST = 2.0              # skip pairs too close
MAX_MEAN_DIST = 8.0              # discard loose constellations
PERIPHERAL_QUANTILE = 0.3        # skip nodes closer to origin than this

DOMAIN_PAIRS = [
    ("ContinuousLinearMap", "Filter",
     "Continuous linear maps <-> Filter convergence"),
    ("LinearMap", "TopologicalSpace",
     "Linear algebra <-> Topological spaces"),
    ("MeasureTheory", "ProbabilityTheory",
     "Measure theory <-> Probability theory"),
    ("Polynomial", "Complex",
     "Polynomials <-> Complex analysis"),
    ("Matroid", "SimpleGraph",
     "Matroids <-> Graph theory"),
    ("Subgroup", "Ideal",
     "Group theory <-> Ideal theory"),
    ("Matrix", "LinearMap",
     "Matrices <-> Linear maps"),
    ("Module", "Ideal",
     "Modules <-> Ideals"),
    ("Algebra", "NumberField",
     "Algebra <-> Number fields"),
    ("Filter", "MeasureTheory",
     "Filters <-> Measure theory"),
    ("Polynomial", "NumberField",
     "Polynomials <-> Number fields"),
    ("CategoryTheory", "AlgebraicGeometry",
     "Category theory <-> Algebraic geometry"),
]

# ── LOAD ───────────────────────────────────────────────────────
print("Loading embeddings...")
checkpoint = torch.load(EMBEDDING_FILE, weights_only=True)
embeddings = checkpoint['embeddings']
names = checkpoint['names']
ball = geoopt.PoincareBall(c=1.0)
norms = torch.norm(embeddings, dim=1)
norm_threshold = torch.quantile(norms, PERIPHERAL_QUANTILE).item()
print(f"  {len(names)} nodes, dim={embeddings.shape[1]}")
print(f"  Peripherality threshold (norm > {norm_threshold:.4f})")


# ── UTILITIES ──────────────────────────────────────────────────

def geodesic_point(u, v, t):
    """Point at parameter t in [0,1] along the Poincare geodesic."""
    t_tensor = torch.tensor(t, dtype=u.dtype)
    direction = ball.mobius_add(-u, v)
    scaled = ball.mobius_scalar_mul(t_tensor, direction)
    return ball.mobius_add(u, scaled)


def get_peripheral_indices(prefix):
    """Get node indices matching prefix AND sufficiently far from origin."""
    return [i for i, n in enumerate(names)
            if n.startswith(prefix) and norms[i].item() >= norm_threshold]


def find_void_on_geodesic(emb_u, emb_v, pool_embeddings):
    """
    Sample points along the geodesic and return the one most
    distant from the nearest node in the POOL (not all nodes).
    """
    best_point = None
    best_min_dist = -1.0

    for t in np.linspace(0.15, 0.85, GEODESIC_POINTS):
        mid = geodesic_point(emb_u, emb_v, float(t))
        dists = ball.dist(mid.unsqueeze(0), pool_embeddings)
        min_dist = dists.min().item()
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_point = mid.detach().clone()

    return best_point, best_min_dist


def extract_constellation(void_pt, pool_indices, pool_embeddings,
                          prefix_a, prefix_b):
    """
    Find the K nearest nodes to the void FROM THE POOL ONLY.
    The pool is domain A ∪ domain B (peripheral nodes only).
    No 'other' nodes can appear.
    """
    dists = ball.dist(void_pt.unsqueeze(0), pool_embeddings).squeeze(0)
    topk_dists, topk_local = torch.topk(dists, k=K_NEIGHBORS, largest=False)

    cluster = []
    count_a = 0
    count_b = 0
    for i in range(K_NEIGHBORS):
        local_idx = topk_local[i].item()
        global_idx = pool_indices[local_idx]
        name = names[global_idx]
        dist = topk_dists[i].item()

        if name.startswith(prefix_a):
            tag = "A"
            count_a += 1
        else:
            tag = "B"
            count_b += 1

        cluster.append((name, dist, tag))

    mean_dist = topk_dists.mean().item()
    return cluster, mean_dist, count_a, count_b


# ── MAIN LOOP ─────────────────────────────────────────────────
all_results = []

for prefix_a, prefix_b, description in DOMAIN_PAIRS:
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  {prefix_a}  <->  {prefix_b}")
    print(f"{'='*60}")

    indices_a = get_peripheral_indices(prefix_a)
    indices_b = get_peripheral_indices(prefix_b)
    print(f"  Domain A ({prefix_a}): {len(indices_a)} peripheral nodes")
    print(f"  Domain B ({prefix_b}): {len(indices_b)} peripheral nodes")

    if len(indices_a) < K_NEIGHBORS or len(indices_b) < K_NEIGHBORS:
        print("  SKIPPED: not enough peripheral nodes")
        continue

    # Build the pool: domain A ∪ domain B (peripheral only)
    pool_indices = indices_a + indices_b
    pool_indices_tensor = torch.tensor(pool_indices, dtype=torch.long)
    pool_embeddings = embeddings[pool_indices_tensor]

    # Track which pool indices are A vs B
    a_set = set(indices_a)

    # ── Sample cross-domain pairs and find voids ──────────────
    voids = []
    for _ in range(PAIRS_TO_SAMPLE):
        ia = np.random.choice(indices_a)
        ib = np.random.choice(indices_b)

        pair_dist = ball.dist(
            embeddings[ia].unsqueeze(0),
            embeddings[ib].unsqueeze(0)
        ).item()

        if pair_dist < MIN_PAIR_DIST:
            continue

        void_pt, void_score = find_void_on_geodesic(
            embeddings[ia], embeddings[ib], pool_embeddings
        )
        voids.append((void_pt, void_score, ia, ib, pair_dist))

    if not voids:
        print("  No voids found")
        continue

    voids.sort(key=lambda x: x[1], reverse=True)
    print(f"  Sampled {len(voids)} voids, best score: {voids[0][1]:.4f}")

    # ── Extract constellations ────────────────────────────────
    reported = 0
    seen = set()

    for void_pt, void_score, anchor_a, anchor_b, pair_dist in voids:
        if reported >= NUM_CONSTELLATIONS:
            break

        cluster, mean_dist, count_a, count_b = extract_constellation(
            void_pt, pool_indices, pool_embeddings,
            prefix_a, prefix_b
        )

        # Filter: tight enough
        if mean_dist > MAX_MEAN_DIST:
            continue

        # Filter: at least MIN_FROM_EACH from each domain
        if count_a < MIN_FROM_EACH or count_b < MIN_FROM_EACH:
            continue

        # Filter: skip duplicates
        key = frozenset(n for n, d, t in cluster)
        if key in seen:
            continue
        seen.add(key)

        reported += 1

        # ── Report ────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"INTERDOMAIN CONSTELLATION {reported} "
              f"(mean dist: {mean_dist:.4f}, "
              f"void: {void_score:.4f})")
        print('='*60)
        print(f"  Geodesic anchors:")
        print(f"    A: {names[anchor_a]}")
        print(f"    B: {names[anchor_b]}")
        print(f"  Pair dist: {pair_dist:.4f}")
        print(f"  Mix: {count_a}A + {count_b}B")
        print(f"  Members:")
        for name, dist, tag in cluster:
            print(f"    [{tag}] {name} (dist: {dist:.4f})")

        # ── LLM Prompt ────────────────────────────────────────
        prompt = (
            f"I am analyzing the Lean 4 Mathlib library.\n"
            f"The following theorems form a tight geometric cluster\n"
            f"straddling two domains in hyperbolic logical space:\n\n"
            f"Domain A: {prefix_a}\n"
            f"Domain B: {prefix_b}\n"
            f"Bridge: {description}\n\n"
        )
        for name, dist, tag in cluster:
            prompt += f"  - {name} [{tag}]\n"
        prompt += (
            f"\nThese sit near a void on the geodesic between the two\n"
            f"domains. Propose a helper lemma or abstraction that\n"
            f"bridges them. Include a sketch in Lean 4 syntax.\n"
        )

        print(f"\nLLM PROMPT:")
        print(prompt)

        all_results.append({
            "description": description,
            "prefix_a": prefix_a,
            "prefix_b": prefix_b,
            "void_score": void_score,
            "mean_dist": mean_dist,
            "count_a": count_a,
            "count_b": count_b,
            "members": [(n, d, t) for n, d, t in cluster],
            "prompt": prompt,
        })

    if reported == 0:
        print("  No mixed constellations found.")

# ── FINAL RANKING ─────────────────────────────────────────────
print(f"\n{'='*60}")
print("FINAL RANKING: ALL INTERDOMAIN CONSTELLATIONS")
print(f"{'='*60}")

if all_results:
    max_void = max(r["void_score"] for r in all_results) or 1
    min_md = min(r["mean_dist"] for r in all_results) or 0.01

    for r in all_results:
        r["composite"] = (
            0.5 * (r["void_score"] / max_void)
            + 0.5 * (min_md / max(r["mean_dist"], 0.01))
        )

    all_results.sort(key=lambda x: x["composite"], reverse=True)

    for rank, r in enumerate(all_results, 1):
        print(f"\n  #{rank} [{r['description']}] "
              f"composite={r['composite']:.4f}")
        print(f"     void={r['void_score']:.3f}, "
              f"mean_dist={r['mean_dist']:.3f}, "
              f"mix={r['count_a']}A+{r['count_b']}B")
        for name, dist, tag in r["members"]:
            print(f"       [{tag}] {name}")

    # Save prompts
    with open("interdomain_prompts.txt", "w") as f:
        for rank, r in enumerate(all_results, 1):
            f.write(f"{'='*60}\n")
            f.write(f"#{rank} [{r['description']}] "
                    f"composite={r['composite']:.4f}\n")
            f.write(f"{'='*60}\n\n")
            f.write(r["prompt"])
            f.write("\n\n")
    print(f"\nPrompts saved to interdomain_prompts.txt")

else:
    print("  No constellations found.")
    print("  Try: lower PERIPHERAL_QUANTILE, raise MAX_MEAN_DIST,")
    print("       or lower MIN_FROM_EACH to 1.")

print(f"\nDone. Found {len(all_results)} interdomain constellations.")