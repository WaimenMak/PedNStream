# -*- coding: utf-8 -*-
# @Time    : 20/01/2025 11:48
# @Author  : mmai
# @FileName: forky_queues
# @Software: PyCharm

import os
import sys
from src.utils.functions import SpeedDensityFd
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import numpy as np
import matplotlib.pyplot as plt
from handlers.output_handler import OutputHandler

# Now you can import using the project structure
from src.utils.visualizer import NetworkVisualizer, progress_callback
# from src.LTM.network import Network
from src.LTM import Network

if __name__ == "__main__":
    # Network configuration

    # adj = np.array([[0, 1, 0, 0, 0, 1],
    #                 [1, 0, 1, 0, 1, 0],
    #                 [0, 1, 0, 1, 0, 0],
    #                 [0, 0, 1, 0, 0, 0],
    #                 [0, 1, 0, 0, 0, 0],
    #                 [1, 0, 0, 0, 0, 0],
    #                 ])

    adj = np.array([[0, 1, 0, 0, 0],
                    [1, 0, 1, 0, 1],
                    [0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0],
                    ])

    '''Scenario 1:'''
    params = {
        'unit_time': 10,
        'simulation_steps': 700,
        'assign_flows_type': 'classic',
        'default_link': {
            'length': 100,
            'width': 1,
            'free_flow_speed': 1.5,
            'k_critical': 2,
            'k_jam': 6,
            'gamma': 0,
            'speed_noise_std': 0.05,
            'fd_type': 'yperman',
        },
        'links': {
            '1_2': {'length': 100, 'width': 1, 'free_flow_speed': 1.5, 'k_critical': 2, 'k_jam': 6,
                    'speed_noise_std': 0.05, 'fd_type': 'yperman', 'controller_type': 'gate'},
            '2_3': {'length': 50, 'width': 1, 'free_flow_speed': 1.5, 'k_critical': 2, 'k_jam': 6, 'speed_noise_std': 0.05, 'fd_type': 'yperman'},
        },
        'demand': {
            "origin_0": {
                "peak_lambda": 15,
                "base_lambda": 5,
            },
            "origin_4": {
                "peak_lambda": 15,
                "base_lambda": 5,
            }
        }
    }
    network_env = Network(adj, params, origin_nodes=[0, 4])
    network_env.update_turning_fractions_per_node(node_ids=[1],
                                                  new_turning_fractions=np.array([[1, 0, 0.5, 0.5, 0, 1]])) #[1_2, 1_4, 1_0, 1_4, 1_0, 1_2]
    # '''Scenario 2:'''
    # params = {
    #     'unit_time': 10,
    #     'simulation_steps': 700,
    #     'assign_flows_type': 'classic',
    #     'k_paths': 1,  # number of paths from origin to destination, if set to 1, it will use the shortest path
    #     'default_link': {
    #         'length': 100,
    #         'width': 1,
    #         'free_flow_speed': 1.5,
    #         'k_critical': 2,
    #         'k_jam': 6,
    #         'gamma': 0,
    #         'speed_noise_std': 0.05,
    #         'fd_type': 'yperman',
    #     },
    #     'demand': {
    #         "origin_3": {
    #             "peak_lambda": 20,
    #             "base_lambda": 5,
    #         },
    #     }
    # }
    # od_flows = {
    #     (3, 0): 10,
    #     (3, 4): 5,
    # }
    # network_env = Network(adj, params, origin_nodes=[3], destination_nodes=[0, 4], od_flows=od_flows)


    network_env.visualize()

    # Run simulation
    network_env.links[(1,2)].front_gate_width = 0.5
    for t in range(1, params['simulation_steps']):
        network_env.network_loading(t)
        if t == 300: # adjust the width of link 2_3, simulate remove the bottleneck
            # pass
            network_env.links[(1,2)].front_gate_width = 2
            # network_env.links[(2,3)].area = 2 * network_env.links[(2,3)].length  # adjust the area of the link to match the new width
            # network_env.links[(2,3)].k_critical = 2
            # network_env.links[(2,3)].k_jam = 6
            # network_env.links[(2,3)].shock_wave_speed = network_env.links[(2,3)].capacity / (6 - 2)
            # network_env.links[(2,3)].speed_density_fd.k_critical = 2
            # network_env.links[(2,3)].speed_density_fd.k_jam = 6




    # Construct paths relative to the project root
    output_dir = os.path.join("..", "outputs")

    # Use the constructed paths
    output_handler = OutputHandler(base_dir=output_dir, simulation_dir="forky_queues")
    output_handler.save_network_state(network_env)

    import matplotlib
    matplotlib.use('macosx')
    visualizer = NetworkVisualizer(simulation_dir=os.path.join(output_dir, "forky_queues"))
    anim = visualizer.animate_network(start_time=0, end_time=params["simulation_steps"], interval=100, edge_property='density', tag=True)

    # # MP4
    # writer = matplotlib.animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'),
    #                                          bitrate=2000)
    # Save the animation as MP4
    # anim.save(os.path.join(output_dir, "forky_queues", f"forky_queues_{params['assign_flows_type']}.mp4"),
    #           writer=writer,
    #           progress_callback=progress_callback)
    plt.show()