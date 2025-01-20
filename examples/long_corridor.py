# -*- coding: utf-8 -*-
# @Time    : 18/01/2025 20:09
# @Author  : mmai
# @FileName: long_corridor
# @Software: PyCharm

# Add project root to Python path
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import numpy as np
import matplotlib.pyplot as plt
from handlers.output_handler import OutputHandler

# Now you can import using the project structure
from src.utils.visualizer import NetworkVisualizer
from src.LTM.network import Network

if __name__ == "__main__":
    # Network configuration

    adj = np.array([[0, 1, 0, 0, 0, 0],
                    [1, 0, 1, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0],
                    [0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 1, 0, 1],
                    [0, 0, 0, 0, 1, 0]])


    params = {
        'length': 100,
        'width': 1,
        'free_flow_speed': 1.5,
        'k_critical': 2,
        'k_jam': 10,
        'unit_time': 10,
        'peak_lambda': 10,
        'base_lambda': 5,
        'simulation_steps': 500,
    }

    # Initialize and run simulation
    network_env = Network(adj, params, od_nodes=[0, 5], origin_nodes=[0, 5])
    network_env.visualize()

    # Run simulation
    for t in range(1, params['simulation_steps']):
        network_env.network_loading(t)

    # # Plot inflow and outflow
    # plt.figure(1)
    # path = [(5, 4), (4, 3), (3, 2), (2, 1), (1, 0)]
    # for link_id in path:
    #     plt.plot(network_env.links[link_id].inflow, label=f'inflow{link_id}')
    #     plt.legend()
    # plt.show()

    # Construct paths relative to the project root
    output_dir = os.path.join("..", "outputs")

    # Use the constructed paths
    output_handler = OutputHandler(base_dir=output_dir, simulation_dir="long_corridor")
    output_handler.save_network_state(network_env)

    import matplotlib
    matplotlib.use('macosx')
    visualizer = NetworkVisualizer(simulation_dir=os.path.join(output_dir, "long_corridor"))
    ani = visualizer.animate_network(start_time=0, end_time=params["simulation_steps"], interval=100, edge_property='density')
    plt.show()