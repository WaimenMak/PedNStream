# -*- coding: utf-8 -*-
# @Time    : 22/01/2025 17:56
# @Author  : mmai
# @FileName: big_network
# @Software: PyCharm

import os
import sys
import json
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from handlers.output_handler import OutputHandler
from src.utils.visualizer import NetworkVisualizer, progress_callback
from src.LTM.network import Network

if __name__ == "__main__":
    # Loading
    with open("../node_positions.json", 'r') as f:
        pos = {str(k): np.array(v) for k, v in json.load(f).items()}

    adj = np.load("../delft_adj_matrix.npy", allow_pickle=False)
    params = {
        'unit_time': 10,
        'simulation_steps': 500,
        'default_link': {
            'length': 50,
            'width': 1,
            'free_flow_speed': 1.5,
            'k_critical': 2,
            'k_jam': 10,
        }
    }

    # Define OD flows
    # od_flows = {
    #     # From node 0
    #     # (0, 4): create_peak_pattern(100),  # Peak pattern to node 4
    #     # (0, 4): 10,
    #     # (0, 5): 5,                        # Constant flow to node 5
    #
    #     # From node 1
    #     # (1, 4): create_pulse_pattern(30, 90, 3, 4),  # Pulse pattern to node 4
    #     (0, 8): 8,
    #     (0, 100): 8,
    #     (5, 8): 8,
    #     (5, 100): 8,
    #     # (1, 5): create_pulse_pattern(40, 80, 5, 3)   # Different pulse to node 5
    #     # (1, 5): 5
    # }
    # Initialize network with origin at node 0 and destination at node 8
    network_env = Network(adj, params, origin_nodes=[0, 5, 20], destination_nodes=[8, 100, 50], pos=pos)
    # Run simulation
    for t in range(1, params['simulation_steps']):
        network_env.network_loading(t)

    # Save and visualize results
    output_dir = os.path.join("..", "outputs")
    output_handler = OutputHandler(base_dir=output_dir, simulation_dir="delft_directions")
    output_handler.save_network_state(network_env)

    # Create animation
    matplotlib.use('macosx')
    visualizer = NetworkVisualizer(simulation_dir=os.path.join(output_dir, "delft_directions"), pos=pos)
    anim = visualizer.animate_network(start_time=0,
                                    end_time=params["simulation_steps"],
                                    # interval=1,
                                    figsize=(14, 12),
                                    edge_property='density')

    # MP4
    writer = matplotlib.animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'),
                                             bitrate=2000)

    # Save the animation as MP4
    anim.save(os.path.join(output_dir, "delft_directions", "delft.mp4"),
              writer=writer,
              progress_callback=progress_callback)
    plt.show()