# Usage Guide for `src/LTM` Folder

This folder contains the core classes and functions for simulating pedestrian traffic using the Link Transmission Model (LTM). Below is a detailed explanation of the key components and how to use them.

---

## Key Classes

### `BaseLink` (`link.py`)
- **Purpose**: Base class for all link types.
- **Attributes**:
  - `link_id`: Unique identifier for the link.
  - `start_node`: Starting node of the link.
  - `end_node`: Ending node of the link.
  - `inflow`: Array to store inflow at each time step.
  - `outflow`: Array to store outflow at each time step.
  - `cumulative_inflow`: Array to store cumulative inflow at each time step.
  - `cumulative_outflow`: Array to store cumulative outflow at each time step.
- **Methods**:
  - `update_cum_outflow(q_j: float, time_step: int)`: Updates the cumulative outflow for the given time step.
  - `update_cum_inflow(q_i: float, time_step: int)`: Updates the cumulative inflow for the given time step.
  - `update_speeds(time_step: int)`: Placeholder method for updating speeds (implemented in subclasses).

### `Link` (`link.py`)
- **Purpose**: Represents a physical link with full traffic dynamics.
- **Attributes**:
  - Inherits all attributes from `BaseLink`.
  - Additional attributes include `length`, `width`, `free_flow_speed`, `capacity`, `k_jam`, `k_critical`, etc.
- **Methods**:
  - `update_link_density_flow(time_step: int)`: Updates the density and flow of pedestrians on the link.
  - `update_speeds(time_step: int)`: Updates the speed of pedestrians on the link based on density.
  - `get_outflow(time_step: int, tau: int) -> int`: Calculates the outflow considering congestion and diffusion.
  - `cal_sending_flow(time_step: int) -> float`: Calculates the sending flow for the link.
  - `cal_receiving_flow(time_step: int) -> float`: Calculates the receiving flow for the link.

---

## Key Functions

### `update_link_density_flow(time_step: int)`
- **Purpose**: Updates the number of pedestrians, density, and flow on the link.
- **Details**:
  - Calculates the change in pedestrian count based on inflow and outflow.
  - Updates the density as the number of pedestrians divided by the link area.
  - Optionally calculates the link flow using either the fundamental diagram or kinematic wave model.

### `update_speeds(time_step: int)`
- **Purpose**: Updates the speed of pedestrians on the link based on density.
- **Details**:
  - Considers the density of the reverse link (if present).
  - Uses the `cal_travel_speed` function to calculate the speed based on the combined density.
  - Updates the travel time and link flow.

---

## Example Usage

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

---

## Key Notes
- **Data Structures**: All dynamic attributes (e.g., `inflow`, `outflow`, `density`) are stored as NumPy arrays for efficient computation.
- **Density and Speed Calculation**: The model uses a combination of density and speed to simulate pedestrian flow dynamics.
- **Path Finding**: The `PathFinder` class (not shown here) is used to find paths between nodes, which is essential for route choice behavior.

