# -*- coding: utf-8 -*-
# @Time    : 21/01/2025 16:11
# @Author  : mmai
# @FileName: nine_node
# @Software: PyCharm

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
from pathlib import Path

if __name__ == "__main__":
    # Create 9x9 adjacency matrix for a grid network
    adj = np.array([
        [0, 1, 0, 1, 0, 0, 0, 0, 0],  # Node 0
        [1, 0, 1, 0, 1, 0, 0, 0, 0],  # Node 1
        [0, 1, 0, 0, 0, 1, 0, 0, 0],  # Node 2
        [1, 0, 0, 0, 1, 0, 1, 0, 0],  # Node 3
        [0, 1, 0, 1, 0, 1, 0, 1, 0],  # Node 4
        [0, 0, 1, 0, 1, 0, 0, 0, 1],  # Node 5
        [0, 0, 0, 1, 0, 0, 0, 1, 0],  # Node 6
        [0, 0, 0, 0, 1, 0, 1, 0, 1],  # Node 7
        [0, 0, 0, 0, 0, 1, 0, 1, 0]   # Node 8
    ])

    # Simulation parameters
    # params = {
    #     'length': 100,
    #     'width': 1,
    #     'free_flow_speed': 1.5,
    #     'k_critical': 2,
    #     'k_jam': 10,
    #     'unit_time': 10,
    #     'peak_lambda': 15,
    #     'base_lambda': 5,
    #     'simulation_steps': 1000,
    # }
    params = {
        'unit_time': 10,
        'simulation_steps': 600,
        'assign_flows_type': 'classic',
        'default_link': {
            'length': 100,
            'width': 1,
            'free_flow_speed': 1.5,
            'k_critical': 2,
            'k_jam': 10,
        },
        'demand':{
            "origin_0":{
                "peak_lambda": 15,
                "base_lambda": 5,
            },
            "origin_4":{
                "peak_lambda": 15,
                "base_lambda": 5,
            }
        }
    }

    # Initialize network with origin at node 0 and destination at node 8
    # network_env = Network(adj, params, origin_nodes=[0, 4]) # if set destination node, just set the demand to be 0
    od_flows = {
        (0, 8): 5,
        (4, 8): 10,
        (0, 3): 5,
        (4, 3): 1,
        (0, 1): 5,
        (4, 1): 1,
    }
    network_env = Network(adj, params, origin_nodes=[0, 4], destination_nodes=[3, 8, 1], od_flows=od_flows)
    network_env.visualize()

    # Run simulation
    for t in range(1, params['simulation_steps']):
        network_env.network_loading(t)

    # Save and visualize results
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "outputs"
    output_handler = OutputHandler(base_dir=str(output_dir), simulation_dir="nine_node")
    output_handler.save_network_state(network_env)

    # Create animation
    matplotlib.use('macosx')
    visualizer = NetworkVisualizer(simulation_dir=os.path.join(output_dir, "nine_node"))
    anim = visualizer.animate_network(start_time=0,
                                    end_time=params["simulation_steps"],
                                    interval=100,
                                    edge_property='density')


    from matplotlib.animation import PillowWriter
    # GIf
    sim_name = "nine_node"
    writer = PillowWriter(fps=15, metadata=dict(artist='Me'))

    # Save the animation with progress tracking
    anim.save(os.path.join(output_dir, f"{sim_name}", "network_animation.gif"),
              writer=writer,
              progress_callback=progress_callback)

    plt.show()

