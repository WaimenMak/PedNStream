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
    with open("../data/delft/node_positions.json", 'r') as f:
        pos = {str(k): np.array(v) for k, v in json.load(f).items()}

    adj = np.load("../data/delft/adj_matrix.npy", allow_pickle=False)
    params = {
        'length': 50,
        'width': 1,
        'free_flow_speed': 1.5,
        'k_critical': 2,
        'k_jam': 10,
        'unit_time': 10,
        'peak_lambda': 15,
        'base_lambda': 5,
        'simulation_steps': 500,
    }

    params = {
        'unit_time': 10,
        'simulation_steps': 500,
        'default_link': {
            'length': 50,
            'width': 1,
            'free_flow_speed': 1.5,
            'k_critical': 2,
            'k_jam': 10,
            },
    }
    # whether adj is adjacent matrix or not
    # print(np.allclose(adj, adj.T))
    # Initialize network with origin at node 0 and destination at node 8
    network_env = Network(adj, params, origin_nodes=[0, 8], pos=pos)
    # network_env.visualize()

    # for node in network_env.nodes:
    #     if node.dest_num == None:
    #         print(node.node_id)
    # Run simulation
    for t in range(1, params['simulation_steps']):
        network_env.network_loading(t)

    # # Save and visualize results
    output_dir = os.path.join("..", "outputs")
    output_handler = OutputHandler(base_dir=output_dir, simulation_dir="delft")
    output_handler.save_network_state(network_env)

    # Create animation
    matplotlib.use('macosx')
    visualizer = NetworkVisualizer(simulation_dir=os.path.join(output_dir, "delft"), pos=pos)
    # visualizer.visualize_network_state(time_step=10)
    anim = visualizer.animate_network(start_time=0,
                                    end_time=params["simulation_steps"],
                                    # interval=1,
                                    figsize=(14, 12),
                                    edge_property='density')

    plt.show()