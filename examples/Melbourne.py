# -*- coding: utf-8 -*-
# @Time    : 27/05/2025 16:34
# @Author  : mmai
# @FileName: Melbourne
# @Software: PyCharm

from src.utils.env_loader import NetworkEnvGenerator
from handlers.output_handler import OutputHandler
from src.utils.visualizer import NetworkVisualizer, progress_callback
import matplotlib.pyplot as plt
import matplotlib
import os
import json
import numpy as np
import matplotlib.animation
import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv("../data/melbourne/melbourne.csv")
    sensor_node_df = pd.read_csv("../data/melbourne/sensor_node_df.csv")
    # Convert minute counts to 10-second intervals (divide counts evenly)
    def expand_to_10sec(minute_counts):
        return np.repeat(minute_counts/6, 6)  # divide by 6 for even distribution
    # set the demand for specific origin nodes
    # Define the function outside with data as a default parameter
    def create_demand_function(data, sensor_node_df):
        def node_demand_from_data(origin_node, params=None, _data=data, _sensor_node_df=sensor_node_df):
            nearest_node = _sensor_node_df[_sensor_node_df["node_id"] == origin_node]["sensor_id"].values[0]
            demand = expand_to_10sec(_data[_data["Location_ID"] == nearest_node]["Direction_1"])
            # get the ceil of the demand
            return np.ceil(demand.values)
        return node_demand_from_data
    
    demand_function = create_demand_function(data, sensor_node_df)
    env_generator = NetworkEnvGenerator()
    network_env = env_generator.create_network('melbourne', [demand_function])


    # Run simulation
    for t in range(1, env_generator.config['params']['simulation_steps']):
        network_env.network_loading(t)
        # adjust the turning fractions of the node, so that the input deamand only flows into one direction
        network_env.update_turning_fractions_per_node(node_ids=[265], new_turning_fractions=np.array([[0.05, 0, 0.95,
                                                                                                0.33, 0.33, 0.33,
                                                                                                0.33, 0.33, 0.33,
                                                                                                0.33, 0.33, 0.33]]))

    # Save and visualize results
    output_dir = os.path.join("..", "outputs")
    output_handler = OutputHandler(base_dir=output_dir, simulation_dir="melbourne")
    output_handler.save_network_state(network_env)

    with open("../data/melbourne/node_positions.json", 'r') as f:
        pos = {str(k): np.array(v) for k, v in json.load(f).items()}
    # Create animation
    matplotlib.use('macosx')
    visualizer = NetworkVisualizer(simulation_dir=os.path.join(output_dir, "melbourne"), pos=pos)
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
    # anim.save(os.path.join(output_dir, "melbourne", f"melbourne.mp4"),
    #           writer=writer,
    #           progress_callback=progress_callback)

    plt.show()