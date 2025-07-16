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
from pathlib import Path
from handlers.output_handler import OutputHandler

# Now you can import using the project structure
from src.utils.visualizer import NetworkVisualizer, progress_callback
from src.LTM.network import Network

if __name__ == "__main__":
    # Network configuration

    adj = np.array([[0, 1, 1, 1, 0, 0],
                    [1, 0, 1, 1, 0, 0],
                    [1, 1, 0, 1, 0, 0],
                    [1, 1, 1, 0, 1, 0],
                    [0, 0, 0, 1, 0, 1],
                    [0, 0, 0, 0, 1, 0]])

    params = {
        'unit_time': 10,
        'simulation_steps': 1200,
        'assign_flows_type': 'classic',
        'custom_pattern': 'spike_pattern', # the name should be the same as the function name
        'default_link': {
            'length': 100,
            'width': 2,
            'free_flow_speed': 1.1,
            'k_critical': 2,
            'k_jam': 6,
            'speed_noise_std': 0.05,
            'fd_type': "greenshields",
            'controller_type': 'gate',  # type of controller
        },
        'demand': {
            'origin_5': {
                'pattern': "spike_pattern",
                'peak_lambda': 20,
                'base_lambda': 10,
            }
        }
    }

    # set the demand pattern for origin 5, 4
    def spike_pattern(origin_id, params):
        # Access nested dictionary values correctly
        peak_lambda = params['demand'][f'origin_{origin_id}']['peak_lambda']
        base_lambda = params['demand'][f'origin_{origin_id}']['base_lambda']
        t = params['simulation_steps']
        time = np.arange(t)

        morning_peak = peak_lambda * np.exp(-(time - t/4)**2 / (2 * (t/20)**2))
        evening_peak = peak_lambda * np.exp(-(time - 3*t/4)**2 / (2 * (t/20)**2))
        lambda_t = base_lambda + morning_peak + evening_peak

        # np.random.seed(seed)

        demand = np.random.poisson(lam=lambda_t)
        demand[200:250] = 30
        demand[300:] = 0
        return demand

    # Initialize and run simulation
    # network_env = Network(adj, params, od_nodes=[5], origin_nodes=[5])
    network_env = Network(adj, params, origin_nodes=[5], demand_pattern=[spike_pattern])
    network_env.visualize()

    
    # network_env.demand_generator.register_pattern('spike', spike_pattern)

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
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "outputs"

    # Use the constructed paths
    simulation_dir = "spike"
    output_handler = OutputHandler(base_dir=str(output_dir), simulation_dir=simulation_dir)
    output_handler.save_network_state(network_env)

    import matplotlib
    matplotlib.use('macosx')
    visualizer = NetworkVisualizer(simulation_dir=os.path.join(output_dir, simulation_dir))
    anim = visualizer.animate_network(start_time=0, interval=100, edge_property='density', tag=True)
    # #  # MP4
    # writer = matplotlib.animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'),
    #                                          bitrate=2000)
    #
    # # Save the animation as MP4
    # anim.save(os.path.join(output_dir, simulation_dir, f"{simulation_dir}_debug.mp4"),
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