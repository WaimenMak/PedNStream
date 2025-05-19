# LTM Pedestrian Simulator - File Structure and Dependencies

## File Structure Overview

The LTM (Link Transmission Model) pedestrian simulator is organized into several key modules that work together to simulate pedestrian movement:

```
src/LTM/
├── __init__.py          # Exports all public classes
├── link.py              # Link classes for modeling physical connections
├── node.py              # Node classes for modeling intersections
├── network.py           # Network class that manages the simulation
├── od_manager.py        # Manages origin-destination flows
├── path_finder.py       # Handles path finding between nodes
└── Usage.md             # This documentation file
```

## Module Dependencies

```
                 ┌───────────────┐
                 │  path_finder  │
                 └───────┬───────┘
                         │
                         ▼
┌───────────┐      ┌───────────┐      ┌───────────────┐
│    link   │◄─────┤  network  │─────►│  od_manager   │
└─────┬─────┘      └─────┬─────┘      └───────────────┘
      │                  │
      │            ┌─────▼─────┐
      └───────────►│    node   │
                   └───────────┘
```

## Module Descriptions and Dependencies

### 1. `link.py`

Contains the `BaseLink` and `Link` classes that model physical connections in the network.

- **Dependencies**: None (base module)
- **Key Classes**:
  - `BaseLink`: Abstract base class for links
  - `Link`: Physical link with full traffic dynamics

### 2. `node.py`

Contains node classes that represent intersections or decision points in the network.

- **Dependencies**: `link.py` (imports `BaseLink`)
- **Key Classes**:
  - `Node`: Base node class
  - `OneToOneNode`: Simple node with one incoming and one outgoing link
  - `RegularNode`: Complex node with multiple incoming and outgoing links

### 3. `network.py`

Core module that manages the entire simulation network.

- **Dependencies**: `node.py`, `link.py`, `od_manager.py`, `path_finder.py`
- **Key Classes**:
  - `Network`: Main class that initializes and runs the simulation

### 4. `od_manager.py`

Manages origin-destination flows and demand patterns.

- **Dependencies**: None (independent module)
- **Key Classes**:
  - `ODManager`: Manages OD flows
  - `DemandGenerator`: Generates demand patterns

### 5. `path_finder.py`

Handles path finding between nodes and calculates turning fractions.

- **Dependencies**: None (operates on network data)
- **Key Classes**:
  - `PathFinder`: Finds paths between origin and destination nodes

## Initialization Flow

1. Create a `Network` instance with adjacency matrix and parameters
2. `Network` initializes nodes and links
3. `ODManager` initializes origin-destination flows
4. `PathFinder` calculates paths between origins and destinations
5. Simulation runs by calling `network_loading` for each time step

## Usage Example

```python
import numpy as np
from src.LTM import Network

# Define network structure
adj_matrix = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 0]
])

# Define parameters
params = {
    'length': 50,
    'width': 2,
    'free_flow_speed': 1.5,
    'k_critical': 2,
    'k_jam': 10,
    'unit_time': 10,
    'simulation_steps': 100
}

# Create network with origin node 0 and destination node 2
network = Network(
    adjacency_matrix=adj_matrix,
    params=params,
    origin_nodes=[0],
    destination_nodes=[2]
)

# Run simulation
for t in range(1, params['simulation_steps']):
    network.network_loading(t)
```

## Fundamental Diagram and Stochastic Movement

The authenticity of the fundamental diagram and stochasticity of pedestrian movement are primarily controlled in:

1. `link.py` - Through the speed-density relationship and flow calculations
2. `node.py` - Through the turning fraction calculations and flow assignment

To improve these aspects, focus on modifying:
- The speed calculation in `Link.update_speeds()`
- The flow calculations in `Link.cal_sending_flow()` and `Link.cal_receiving_flow()`
- The turning fraction logic in `PathFinder.calculate_node_turning_fractions()`

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

