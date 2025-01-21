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
        'peak_lambda': 15,
        'base_lambda': 5,
        'simulation_steps': 800,
    }

    # Initialize and run simulation
    network_env = Network(adj, params, od_nodes=[0, 5], origin_nodes=[0, 5])
    network_env.visualize()

    # Run simulation
    for t in range(1, params['simulation_steps']):
        network_env.network_loading(t)

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
    output_dir = os.path.join("..", "outputs")

    # Use the constructed paths
    output_handler = OutputHandler(base_dir=output_dir, simulation_dir="long_corridor")
    output_handler.save_network_state(network_env)

    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter
    from tqdm import tqdm
    matplotlib.use('macosx')

    # Create the visualization
    visualizer = NetworkVisualizer(simulation_dir=os.path.join(output_dir, "long_corridor"))
    anim = visualizer.animate_network(start_time=0, end_time=params["simulation_steps"],
                                    interval=100, edge_property='density')

    # # Set up the writer
    # writer = PillowWriter(fps=15, metadata=dict(artist='Me'))

    # # Save the animation with progress tracking
    # anim.save(os.path.join(output_dir, "long_corridor", "network_animation.gif"),
    #           writer=writer,
    #           progress_callback=progress_callback)

    # plt.show()

    writer = matplotlib.animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'), 
                                             bitrate=2000)

    # Save the animation as MP4
    anim.save(os.path.join(output_dir, "long_corridor", "network_animation.mp4"),
              writer=writer,
              progress_callback=progress_callback)

    plt.show()
