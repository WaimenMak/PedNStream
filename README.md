# PedNStream: Pedestrian Network Flow Simulator

PedNStream is a light-weight/Python-native pedestrian traffic simulation tool based on the Link Transmission Model (LTM). It enables modeling and simulation of pedestrian movements through complex networks, providing insights into crowd dynamics and flow behaviors.

## Key Features

- **Network-based Simulation**: Model pedestrian movement through interconnected pathways and decision points
- **Flexible Network Configuration**: Support for various network topologies from simple corridors to complex urban layouts
<!-- - **Dynamic Flow Modeling**: Incorporates traffic dynamics including densities, speeds, and queue formation -->
- **Customizable Parameters**: Configure link properties, flow speeds, and capacity constraints
- **Visualization Tools**: Analyze simulation outputs through interactive visualizations and animations

## Core Components

### Network Module (`src/LTM/network.py`)
The central component managing the simulation network and execution:
- Network initialization and state management
- Step-by-step simulation execution
- Integration of links and nodes

### Link Module (`src/LTM/link.py`)
Models physical pathways with properties like:
- Length and width
- Free-flow speed
- Critical and jam densities
- Bi-directional flow support

### Node Module (`src/LTM/node.py`)
Handles intersection points and decision making:
- Flow distribution between connected links
- Turn probability management
- Origin-Destination (OD) pair handling

### Path Finding (`src/LTM/path_finder.py`)
Manages route choice and navigation:
- Path discovery between origins and destinations
- Turn probability calculations
- Distance-based routing

## Quick Start

### Basic Network Simulation

```python
from src.utils.env_loader import NetworkEnvGenerator
from src.utils.visualizer import NetworkVisualizer

# Initialize network from configuration
env_generator = NetworkEnvGenerator()
network_env = env_generator.create_network("example_config")

# Run simulation
for t in range(1, env_generator.config['params']['simulation_steps']):
    network_env.network_loading(t)

# Visualize results
visualizer = NetworkVisualizer(simulation_dir="output_dir")
anim = visualizer.animate_network(
    start_time=0,
    end_time=env_generator.config['params']['simulation_steps'],
    edge_property='density'
)
```

### Configuration Example

```yaml
# sim_params.yaml
params:
  unit_time: 10
  simulation_steps: 500
  default_link:
    length: 50
    width: 1
    free_flow_speed: 1.5
    k_critical: 2
    k_jam: 10
  demand:
    origin_0:
      peak_lambda: 15
      base_lambda: 5
```

## Example Scenarios

The `examples/` directory contains various simulation scenarios:
- `long_corridor.py`: Simple bi-directional flow in a corridor
- `nine_node.py`: Grid network with multiple OD pairs
- `delft_exp.py`: Real-world network simulation (Delft, Netherlands)
- `melbourne.py`: Large-scale urban network simulation

## Visualization

PedNStream provides rich visualization capabilities:
- Network state visualization
- Density and flow animations
- Interactive network dashboard
- Time-series analysis tools

![Network Animation Example](./README.assets/network_animation.gif)

## Project Structure

```
project_root/
├── src/
│   ├── LTM/              # Core simulation components
│   │   ├── link.py       # Link dynamics
│   │   ├── node.py       # Node behavior
│   │   ├── network.py    # Network management
│   │   └── path_finder.py # Route choice
│   └── utils/            # Utility functions
│       ├── visualizer.py # Visualization tools
│       └── env_loader.py # Environment setup
├── examples/             # Example scenarios
└── data/                 # Network configurations
```

## Contributing

Contributions are welcome! Please check the issues page for current development priorities.

## License

PedNStream is released under the MIT License. See the [LICENSE](./LICENSE) file for details.