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
from src.utils.visualizer import NetworkVisualizer, progress_callback
from src.LTM.network import Network
from pathlib import Path

if __name__ == "__main__":
    # Network configuration

    adj = np.array([[0, 1, 0, 0, 0, 0],
                    [1, 0, 1, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0],
                    [0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 1, 0, 1],
                    [0, 0, 0, 0, 1, 0]])

    # Initialize and run simulation
    # ''' Scenario 1 '''
    params = {
        'unit_time': 10,
        'simulation_steps': 600,
        'default_link': {
            'length': 100,  # make it to 50 to see spillback
            'width': 2,
            'free_flow_speed': 1.1,
            'k_critical': 2,
            'k_jam': 6,
            'fd_type': 'yperman',  # type of fundamental diagram
            'bi_factor': 1,  # factor for bi-directional FD
            'controller_type': 'gate',  # type of controller
        },
        'demand': {
            "origin_0": {
                "peak_lambda": 25,
                "base_lambda": 5,
            },
            "origin_5": {
                "peak_lambda": 25,
                "base_lambda": 5,
            }
        }

    }
    network_env = Network(adj, params, origin_nodes=[5, 0])
    # # # Set demand for nodes to zero after a certain time step
    # network_env.nodes[0].demand[300:] = np.zeros(600 - 300)
    # network_env.nodes[5].demand[300:] = np.zeros(600 - 300)
    # ''' Scenario 2 '''
    # params = {
    #     'unit_time': 10,
    #     'simulation_steps': 1200,
    #     'assign_flows_type': 'classic',
    #     'default_link': {
    #         'length': 50,  # make it to 50 to see spillback
    #         'width': 2,
    #         'free_flow_speed': 1.1,
    #         'k_critical': 1,
    #         'k_jam': 6,
    #         'activity_probability': 0,  # probability of activity on the link
    #         'fd_type': 'yperman',  # type of fundamental diagram
    #         'bi_factor': 1,  # factor for bi-directional FD
    #         'speed_noise_std': 0,  # whether to add noise to the speed
    #         'controller_type': 'gate',  # type of controller
    #     },
    #     'links': {
    #         '2_3': {
    #             'length': 50,
    #             'width': 1,
    #             'free_flow_speed': 1.1,
    #             'k_critical': 2,
    #             'k_jam': 6,
    #             'activity_probability': 0,  # probability of activity on the link
    #             'fd_type': 'yperman',  # type of fundamental diagram
    #             'bi_factor': 1,  # factor for bi-directional FD
    #             'speed_noise_std': 0,  # whether to add noise to the speed
    #             'controller_type': 'gate',  # type of controller
    #         },
    #     },
    #     'demand': {
    #         "origin_3": {
    #             "peak_lambda": 20,
    #             "base_lambda": 8,
    #         },
    #         "origin_2": {
    #             "peak_lambda": 20,
    #             "base_lambda": 8,
    #         }
    #     }
    #
    # }
    # network_env = Network(adj, params, origin_nodes=[2, 3])
    # network_env.update_turning_fractions_per_node(node_ids=[2, 3],
    #                                                       new_turning_fractions=np.array([[0, 1, 0.5, 0.5, 0, 1],
    #                                                                                       [1, 0, 0, 1, 0.5, 0.5]]))
    # Set demand for nodes to zero after a certain time step
    # network_env.nodes[2].demand[0:10] = np.zeros(10)
    # network_env.nodes[3].demand[40:] = np.zeros(1200 - 40)
    # network_env.nodes[2].demand[80:100] = np.zeros(20)
    # network_env.nodes[3].demand[30:] = np.zeros(600 - 30)

    network_env.visualize()


    # Run simulation
    # network_env.links[(3,4)].back_gate_width = 0
    # network_env.links[(2,1)].back_gate_width = 0
    for t in range(1, params['simulation_steps']):
        network_env.network_loading(t)
        # if t == 120:
            # network_env.links[(3,4)].back_gate_width = 1
            # network_env.links[(2,1)].back_gate_width = 0
        # if t == 32:
        #     network_env.links[(3,4)].back_gate_width = 0
        #     network_env.links[(2,1)].back_gate_width = 0
        #     network_env.nodes[0].demand[201:251] = np.random.poisson(lam=np.ones(50) * 10, size=50)

    # # Plot inflow and outflow
    # plt.figure(1)
    # path = [(2, 3)]
    # for link_id in path:
    #     plt.plot(network_env.links[link_id].inflow, label=f'inflow{link_id}')
    #     plt.plot(network_env.links[link_id].outflow, label=f'outflow{link_id}')
    #     plt.legend()
    # plt.show()

    # plt.figure(2)
    # for link_id in path:
    #     plt.plot(network_env.links[link_id].cumulative_inflow, label=f'cumulative_inflow{link_id}')
    #     plt.plot(network_env.links[link_id].cumulative_outflow, label=f'cumulative_outflow{link_id}')
    #     plt.legend()
    # plt.show()

    # Construct paths relative to the project root
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "outputs"

    sim_name = "long_corridor"
    # Use the constructed paths
    output_handler = OutputHandler(base_dir=str(output_dir), simulation_dir=f"{sim_name}")
    output_handler.save_network_state(network_env)

    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter
    from tqdm import tqdm
    matplotlib.use('macosx')

    # Create the visualization
    visualizer = NetworkVisualizer(simulation_dir=os.path.join(output_dir, f"{sim_name}"))
    anim = visualizer.animate_network(start_time=0, end_time=params["simulation_steps"],
                                    interval=100, edge_property='density', tag=True)

    # GIf
    writer = PillowWriter(fps=8, metadata=dict(artist='Me'))

    # Save the animation with progress tracking
    anim.save(os.path.join(output_dir, f"{sim_name}", "network_animation.gif"),
              writer=writer,
              progress_callback=progress_callback)

    # plt.show()
    
    # MP4
    # writer = matplotlib.animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'),
    #                                          bitrate=2000)
    #
    # # Save the animation as MP4
    # anim.save(os.path.join(output_dir, "long_corridor", "network_animation.mp4"),
    #           writer=writer,
    #           progress_callback=progress_callback)

    plt.show()
