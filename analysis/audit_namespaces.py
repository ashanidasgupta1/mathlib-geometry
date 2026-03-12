"""
audit_namespaces.py
"""

import torch
from collections import Counter

EMBEDDING_FILE = "mathlib_hnn_embeddings.pt"

print("Loading embeddings...")
checkpoint = torch.load(EMBEDDING_FILE, weights_only=True)
names = checkpoint['names']
print(f"  {len(names)} total declarations\n")

# ── Level 1 prefixes (e.g. "MeasureTheory", "Topology") ──────
print("=" * 60)
print("TOP-LEVEL PREFIXES (first component before '.')")
print("=" * 60)
level1 = Counter()
for n in names:
    level1[n.split(".")[0]] += 1

for prefix, count in level1.most_common(50):
    print(f"  {count:>6d}  {prefix}")

# ── Level 2 prefixes (e.g. "MeasureTheory.Measure") ──────────
print(f"\n{'=' * 60}")
print("LEVEL-2 PREFIXES (first two components)")
print("=" * 60)
level2 = Counter()
for n in names:
    parts = n.split(".")
    key = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
    level2[key] += 1

for prefix, count in level2.most_common(80):
    print(f"  {count:>6d}  {prefix}")

# ── Interactive search ────────────────────────────────────────
print(f"\n{'=' * 60}")
print("SEARCH: type a prefix to see how many nodes match")
print("(empty line to quit)")
print("=" * 60)
while True:
    query = input("\n  prefix> ").strip()
    if not query:
        break
    count = sum(1 for n in names if n.startswith(query))
    print(f"  {count} nodes match '{query}'")
    if count > 0 and count <= 20:
        for n in names:
            if n.startswith(query):
                print(f"    {n}")
    elif count > 20:
        matches = [n for n in names if n.startswith(query)]
        for n in matches[:10]:
            print(f"    {n}")
        print(f"    ... and {count - 10} more")
