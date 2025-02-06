# -*- coding: utf-8 -*-
# @Time    : 26/11/2024 12:10
# @Author  : mmai
# @FileName: draft
# @Software: PyCharm

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import numpy as np
import matplotlib.pyplot as plt
from handlers.output_handler import OutputHandler

# Now you can import using the project structure
from src.utils.visualizer import NetworkVisualizer, progress_callback
from src.LTM.network import Network

if __name__ == "__main__":
    # Network configuration
    # adj = np.array([[0, 1],
    #                 [1, 0]])
    
    # adj = np.array([[0, 1, 1, 1],
    #                 [1, 0, 1, 1],
    #                 [1, 1, 0, 1],
    #                 [1, 1, 1, 0]])

    adj = np.array([[0, 1, 1, 1, 0, 0],
                    [1, 0, 1, 1, 0, 0],
                    [1, 1, 0, 1, 0, 0],
                    [1, 1, 1, 0, 1, 0],
                    [0, 0, 0, 1, 0, 1],
                    [0, 0, 0, 0, 1, 0]])

    # adj = np.array([[0, 0, 0, 1, 0, 0],
    #                 [0, 0, 0, 1, 0, 0],
    #                 [0, 0, 0, 1, 0, 0],
    #                 [1, 1, 1, 0, 1, 0],
    #                 [0, 0, 0, 1, 0, 1],
    #                 [0, 0, 0, 0, 1, 0]])


    params = {
        'length': 100,
        'width': 1,
        'free_flow_speed': 1.5,
        'k_critical': 2,
        'k_jam': 10,
        'unit_time': 10,
        'peak_lambda': 10,
        'base_lambda': 5,
        'simulation_steps': 1000,
    }

    # Initialize and run simulation
    # network_env = Network(adj, params, od_nodes=[5], origin_nodes=[5])
    network_env = Network(adj, params, origin_nodes=[5])
    network_env.visualize()

    # Run simulation
    for t in range(1, params['simulation_steps']):
        # if t == 1:
        #     network_env.update_turning_fractions_per_node(node_ids=[3],
        #                                                    new_turning_fractions=np.array([[0.33, 0.33, 0.33, 1, 0, 0, 1, 0, 0, 0.33333, 0.33333, 0.33333]]))  # [0_1, 0_5]
        network_env.network_loading(t)
    
    # Plot inflow and outflow
    # plt.figure(1)
    # path = [(5, 4), (4, 3), (3, 2), (2, 1), (1, 0)]
    # for link_id in path:
    #     plt.plot(network_env.links[link_id].inflow, label=f'inflow{link_id}')
    #     plt.legend()
    # plt.show()

    # Construct paths relative to the project root
    output_dir = os.path.join("..", "outputs")

    # Use the constructed paths
    simulation_dir = "spike"
    output_handler = OutputHandler(base_dir=output_dir, simulation_dir=simulation_dir)
    output_handler.save_network_state(network_env)

    import matplotlib
    matplotlib.use('macosx')
    visualizer = NetworkVisualizer(simulation_dir=os.path.join(output_dir, simulation_dir))
    anim = visualizer.animate_network(start_time=0, interval=100, edge_property='density')
    # #  # MP4
    # writer = matplotlib.animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'),
    #                                          bitrate=2000)
    #
    # # Save the animation as MP4
    # anim.save(os.path.join(output_dir, simulation_dir, f"{simulation_dir}.mp4"),
    #           writer=writer,
    #           progress_callback=progress_callback)
    plt.show()


# # Plot density and speed
# plt.figure(2)
# plt.plot(network.links[link_id].density, label='density')
# plt.plot(network.links[link_id].speed, label='speed')
# plt.legend()
# plt.show()

# # Plot cumulative flows
# plt.figure(3)
# plt.plot(network.links[link_id].cumulative_inflow, label='cumulative_inflow')
# plt.plot(network.links[link_id].cumulative_outflow, label='cumulative_outflow')
# plt.legend()
# plt.show()