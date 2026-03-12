import torch
import geoopt
import numpy as np
import os

# --- PREPARATION ---
# Check if the embedding file exists
if not os.path.exists("mathlib_hnn_embeddings.pt"):
    print("Error: mathlib_hnn_embeddings.pt not found. Please run train_hnn.py first.")
    exit()

print("Loading Mathlib Hyperbolic Embeddings...")
checkpoint = torch.load("mathlib_hnn_embeddings.pt")
embeddings = checkpoint['embeddings']
names = checkpoint['names']
name_to_id = checkpoint['name_to_id']

# Initialize the Poincare Ball manifold
ball = geoopt.PoincareBall(c=1.0)

def get_closest_theorems(point, top_k=5):
    """
    Calculates the hyperbolic distance from a point to all 440k theorems
    and returns the top_k closest matches.
    """
    # point is [1, DIM], embeddings is [N, DIM]
    dists = ball.dist(point, embeddings)
    closest_indices = torch.topk(dists, k=top_k, largest=False).indices
    return [(names[i], dists[i].item()) for i in closest_indices]

def predict_geodesic_path(start_name, end_name, steps=12):
    """
    Computes the hyperbolic geodesic arc between two theorems and
    identifies the 'milestone' theorems along that path.
    """
    if start_name not in name_to_id or end_name not in name_to_id:
        print(f"Error: One of the names provided is not in Mathlib. Check spelling/capitalization.")
        return

    # Get vectors and add batch dimension
    u = embeddings[name_to_id[start_name]].unsqueeze(0)
    v = embeddings[name_to_id[end_name]].unsqueeze(0)

    print(f"\n--- Geodesic Discovery: {start_name} → {end_name} ---")
    
    # Generate points along the curved geodesic arc
    t_vals = torch.linspace(0, 1, steps).view(-1, 1)
    geodesic_points = ball.geodesic(t_vals, u, v)

    path = []
    print(f"{'STEP / THEOREM':<50} | {'DIST TO ARC'}")
    print("-" * 65)

    for i in range(steps):
        p_t = geodesic_points[i].unsqueeze(0)
        
        # Find the single closest theorem to this specific point on the arc
        closest_name, distance = get_closest_theorems(p_t, top_k=1)[0]
        
        # Filter duplicates to show progress
        if not path or path[-1] != closest_name:
            path.append(closest_name)
            print(f"{closest_name:<50} | {distance:.4f}")
    
    return path

# --- INTERACTIVE INTERFACE ---
if __name__ == "__main__":
    print("\nMathematical Knowledge Graph AI Loaded.")
    print("Coordinates: 16-Dimensional Poincare Ball")
    print("Hyperbolicity (Delta): 0.1727")
    print("-" * 30)
    print("Enter 'exit' at any time to quit.")

    while True:
        print("\nEnter two theorem names (e.g., 'Nat.add_assoc')")
        start = input("From: ").strip()
        if start.lower() == 'exit': break
        
        end = input("To:   ").strip()
        if end.lower() == 'exit': break
        
        try:
            predict_geodesic_path(start, end)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")