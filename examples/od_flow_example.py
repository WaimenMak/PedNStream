import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from handlers.output_handler import OutputHandler
from src.utils.visualizer import NetworkVisualizer, progress_callback
from src.LTM.network import Network

"""
Create a simple network with 6 nodes:

0 --- 2 --- 4
|     |     |
1 --- 3 --- 5

With flows from nodes 0,1 to nodes 4,5
"""
# Create adjacency matrix (6x6)
adj = np.array([
    [0, 1, 1, 0, 0, 0],  # node 0
    [1, 0, 0, 1, 0, 0],  # node 1
    [1, 0, 0, 1, 1, 0],  # node 2
    [0, 1, 1, 0, 0, 1],  # node 3
    [0, 0, 1, 0, 0, 1],  # node 4
    [0, 0, 0, 1, 1, 0]   # node 5
])

# Simulation parameters
params = {
    'length': 100,
    'width': 1,
    'free_flow_speed': 1.5,
    'k_critical': 2,
    'k_jam': 10,
    'unit_time': 10,
    'peak_lambda': 15,
    'base_lambda': 5,
    'simulation_steps': 500,
}

# Define OD flows with different patterns
simulation_steps = params['simulation_steps']

# Create time-varying demand patterns
def create_peak_pattern(max_flow):
    """Create a peak pattern that rises and falls with integer values"""
    half_steps = simulation_steps // 2
    # Generate integer values for rise and fall
    rise = np.arange(0.5 * max_flow, max_flow, (max_flow - 0.5 * max_flow) / (half_steps - 1))
    rise = np.round(rise).astype(int)

    fall = np.arange(max_flow, 0.5 * max_flow, -(max_flow - 0.5 * max_flow) / (simulation_steps - half_steps - 1))
    fall = np.round(fall).astype(int)

    return np.concatenate([rise, fall])

def create_pulse_pattern(base_flow, pulse_flow, pulse_start, pulse_duration):
    """Create a pattern with a temporary pulse"""
    flow = np.ones(simulation_steps) * base_flow
    flow[pulse_start:pulse_start + pulse_duration] = pulse_flow
    return flow

# Define OD flows
od_flows = {
    # From node 0
    # (0, 4): create_peak_pattern(100),  # Peak pattern to node 4
    # (0, 4): 10,
    # (0, 5): 5,                        # Constant flow to node 5

    # From node 1
    # (1, 4): create_pulse_pattern(30, 90, 3, 4),  # Pulse pattern to node 4
    (1, 4): 8,
    # (1, 5): create_pulse_pattern(40, 80, 5, 3)   # Different pulse to node 5
    (1, 5): 5
}

# Initialize network with origin at node 0 and destination at node 8
network_env = Network(adj, params, origin_nodes=[1], destination_nodes=[4, 5], od_flows=od_flows) # if set destination node, just set the demand to be 0
# network_env.visualize()

# Run simulation
for t in range(1, params['simulation_steps']):
    network_env.network_loading(t)


# Save and visualize results
output_dir = os.path.join("..", "outputs")
output_handler = OutputHandler(base_dir=output_dir, simulation_dir="path_finder")
output_handler.save_network_state(network_env)

# Create animation
matplotlib.use('macosx')
visualizer = NetworkVisualizer(simulation_dir=os.path.join(output_dir, "path_finder"))
anim = visualizer.animate_network(start_time=0,
                                end_time=params["simulation_steps"],
                                interval=100,
                                edge_property='density')

plt.show()