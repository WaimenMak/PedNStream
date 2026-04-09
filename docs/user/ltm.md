
# LTM Package

This package contains the core classes and functions for simulating pedestrian traffic using the Link Transmission Model (LTM). Below is a detailed explanation of the key components and how to use them.

## Examples

### 1. **Initializing a Network**
```python
from src.LTM.network import Network
from src.LTM.link import Link

# Create adjacency matrix
adj = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
])

# Create network
params = {
    'unit_time': 10,
    'simulation_steps': 700,
    'default_link': {
        'length': 100,
        'width': 1,
        'free_flow_speed': 1.5,
        'k_critical': 2,
        'k_jam': 10,
    },
    'demand': {
        "origin_0": {
            "peak_lambda": 25,
            "base_lambda": 5,
        },
        "origin_4": {
            "peak_lambda": 25,
            "base_lambda": 5,
        }
    }
}

network = Network(adj, params)
```

### 2. **Running the Simulation**
```python
for t in range(1, params['simulation_steps']):
    network.network_loading(t)
```

### 3. **Visualizing Results**
```python
network.visualize(figsize=(12, 12), node_size=800, edge_width=2,
                  show_labels=True, label_font_size=12, alpha=0.8)
```
