import torch
import torch.nn as nn
import geoopt
import json
import random
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
DIM = 16            # Embedding dimensions
CURVATURE = 1.0     # Curvature of the Poincare Ball
LR = 0.3            # Higher learning rate is common for Riemannian optimization
EPOCHS = 1
BATCH_SIZE = 2048   # Large batch for speed
NEG_SAMPLES = 5    # Fake edges per real edge

class PoincareEmbedding(nn.Module):
    def __init__(self, num_nodes, dim):
        super().__init__()
        self.ball = geoopt.PoincareBall(c=CURVATURE)
        # Initialize nodes near the center of the ball
        init_data = self.ball.random(num_nodes, dim) * 0.01
        self.embeddings = geoopt.ManifoldParameter(init_data, manifold=self.ball)

    def forward(self, u_idx, v_idx):
        u = self.embeddings[u_idx]
        v = self.embeddings[v_idx]
        return self.ball.dist(u, v)

# --- DATA LOADING ---
print("Loading graph data...")
with open("mathlib_graph.json", 'r') as f:
    data = json.load(f)

# Map names to IDs and build edge list
names = [entry['name'] for entry in data]
name_to_id = {name: i for i, name in enumerate(names)}
edges = []
for entry in data:
    u = name_to_id[entry['name']]
    for v_name in entry['edges']:
        if v_name in name_to_id:
            edges.append((u, name_to_id[v_name]))

print(f"Loaded {len(names)} nodes and {len(edges)} edges.")

# --- TRAINING SETUP ---
model = PoincareEmbedding(len(names), DIM)
optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=LR)

def train():
    model.train()
    random.shuffle(edges)
    
    for i in range(0, len(edges), BATCH_SIZE):
        batch_edges = edges[i:i + BATCH_SIZE]
        if not batch_edges: continue
        
        # Real edges
        u_pos = torch.tensor([e[0] for e in batch_edges], dtype=torch.long)
        v_pos = torch.tensor([e[1] for e in batch_edges], dtype=torch.long)
        
        # Fake edges
        u_neg = u_pos.repeat_interleave(NEG_SAMPLES)
        v_neg = torch.randint(0, len(names), (len(u_pos) * NEG_SAMPLES,), dtype=torch.long)
        
        optimizer.zero_grad()
        
        # Distances
        pos_dist = model(u_pos, v_pos)
        neg_dist = model(u_neg, v_neg)
        
        # FIX: Reshape pos_dist to match the negative samples
        pos_dist_expanded = pos_dist.repeat_interleave(NEG_SAMPLES)
        
        # Log-sigmoid loss
        loss = -torch.log(torch.sigmoid(neg_dist - pos_dist_expanded) + 1e-6).mean()
        
        loss.backward()
        optimizer.step()
        
        if i % (BATCH_SIZE * 20) == 0:
            print(f"Progress: {i}/{len(edges)} | Loss: {loss.item():.4f}")

# --- EXECUTION WITH AUTO-SAVE ---
try:
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train()
except KeyboardInterrupt:
    print("\n[User Interrupt] Catching Ctrl+C... stopping training.")
finally:
    # This block runs whether you finish all epochs OR press Ctrl+C
    print("Saving current embeddings to mathlib_hnn_embeddings.pt...")
    torch.save({
        'embeddings': model.embeddings.data,
        'names': names,
        'name_to_id': name_to_id
    }, "mathlib_hnn_embeddings.pt")
    print("Done! File saved successfully.")