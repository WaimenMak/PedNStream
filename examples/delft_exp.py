# -*- coding: utf-8 -*-
# @Time    : 13/03/2025 11:50
# @Author  : mmai
# @FileName: delft_yaml
# @Software: PyCharm

"""
This example use the real street network of Delft, Netherlands.
"""


from src.utils.env_loader import NetworkEnvGenerator
from handlers.output_handler import OutputHandler
from src.utils.visualizer import NetworkVisualizer, progress_callback
import matplotlib.pyplot as plt
import matplotlib
import os
import json
import numpy as np
import matplotlib.animation


if __name__ == "__main__":
    env_generator = NetworkEnvGenerator()
    network_env = env_generator.create_network("delft") # delft is the name of the file in the data folder including the .yaml config file, edge distances, adjacency matrix, node positions
    # Run simulation
    for t in range(1, env_generator.config['params']['simulation_steps']):
        network_env.network_loading(t)

    # Save and visualize results
    output_dir = os.path.join("..", "outputs")
    output_handler = OutputHandler(base_dir=output_dir, simulation_dir="delft_exp")
    output_handler.save_network_state(network_env)

    with open("../data/delft/node_positions.json", 'r') as f:
        pos = {str(k): np.array(v) for k, v in json.load(f).items()}
    # Create animation
    matplotlib.use('macosx')
    visualizer = NetworkVisualizer(simulation_dir=os.path.join(output_dir, "delft_exp"), pos=pos)
    anim = visualizer.animate_network(start_time=0,
                                    end_time=env_generator.config['params']['simulation_steps'],
                                    # interval=1,
                                    figsize=(14, 12),
                                    edge_property='density')
    # MP4
    # writer = matplotlib.animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'),
    #                                          bitrate=2000)
    #
    # # Save the animation as MP4
    # anim.save(os.path.join(output_dir, "delft_exp", f"delft_exp2.mp4"),
    #           writer=writer,
    #           progress_callback=progress_callback)
    
    plt.show()