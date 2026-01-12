import numpy as np
import json
size = 7
n_nodes = size * size
adj_matrix = np.zeros((n_nodes, n_nodes), dtype=int)

for r in range(size):
    for c in range(size):
        curr = r * size + c
        
        # Check Right
        if c < size - 1:
            right = r * size + (c + 1)
            adj_matrix[curr][right] = 1
            adj_matrix[right][curr] = 1
            
        # Check Down
        if r < size - 1:
            down = (r + 1) * size + c
            adj_matrix[curr][down] = 1
            adj_matrix[down][curr] = 1

positions = {}

for i in range(size * size):
    x = i % size      # Column index
    y = i // size     # Row index
    positions[str(i)] = (x, y)

with open('45_intersections/node_positions.json', 'w') as f:
    json.dump(positions, f)
with open('45_intersections/adj_matrix.npy', 'wb') as f:
    np.save(f, adj_matrix)