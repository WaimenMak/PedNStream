# -*- coding: utf-8 -*-
# @Time    : 19/04/2026
# @Author  : mmai
# @FileName: Melbourne_validation
# @Software: PyCharm

"""
Melbourne Validation Experiment
===============================
Uses boundary nodes as OD with data-driven demand, then compares
simulated flows at interior sensor locations against observed counts.

This script:
1. Loads count data for ALL boundary + interior sensor nodes
2. Sets demand for boundary OD nodes from sensor data
3. Runs the LTM simulation
4. Saves network state for post-processing by validation_analysis.py
"""

from pednstream.utils.network_env_generator import NetworkEnvGenerator
from pednstream.utils.output_handler import OutputHandler
from pednstream.utils.config import load_config
import numpy as np
import pandas as pd
import os


def expand_to_10sec(minute_counts):
    """Convert minute counts to 10-second intervals (divide counts evenly)."""
    return np.repeat(minute_counts / 6, 6)


def create_demand_function(data, sensor_node_df):
    """
    Create a demand function that maps origin nodes to their sensor counts.
    
    For each origin node, finds the nearest sensor and uses its directional
    count data as the demand input. Direction is chosen based on the node's
    likely outflow direction.
    
    Boundary-node-to-sensor mapping based on geographical edge-distance:
      Node 183 -> Sensor 5   (Dir_1: North = into network)
      Node 84  -> Sensor 136 (Dir_1)
      Node 85  -> Sensor 75  (Dir_1)
      Node 150 -> Sensor 31  (Dir_1)
      Node 258 -> Sensor 17  (Dir_1)
      Node 40  -> Sensor 179 (Dir_1)
    """
    # Manual mapping: origin_node -> (sensor_id, direction_column)
    # We found these nodes by filtering for outer geographical bounds with sensors.
    node_sensor_map = {
        51:  (8, "Direction_1"),
        54:  (11, "Direction_1"),
        86:  (140, "Direction_1"),
        92:  (136, "Direction_2"),
        93:  (75, "Direction_1"),
        137: (43, "Direction_2"),
        186: (25, "Direction_1"),
        255: (50, "Direction_2"),
        262: (17, "Direction_2"),
        321: (130, "Direction_2"),
        163: (86, "Direction_2"),
    }

    # node_sensor_map = {
    #     51:  (8, "Direction_1"),
    #     54:  (11, "Direction_1"),
    #     86:  (140, "Direction_1"),
    #     92:  (136, "Direction_2"),
    #     93:  (75, "Direction_1"),
    #     137: (43, "Direction_2"),
    #     186: (25, "Direction_1"),
    #     255: (50, "Direction_2"),
    #     262: (17, "Direction_2"),
    #     321: (130, "Direction_2"),
    #     163: (86, "Direction_2"),
    #     288: (79, "Direction_1"),
    #     309: (69, "Direction_1"),
    #     292: (2, "Direction_1"),
    #     41: (179, "Direction_1"),
    #     302: (63, "Direction_1"),
    #     196: (6, "Direction_1"),
    #     272: (61, "Direction_1"),
    #     310: (52, "Direction_1"),
    #     292: (1, "Direction_1"),
    #     189: (118, "Direction_1"),
    #     330: (65, "Direction_1"),
    #     340: (37, "Direction_1"),
    #     294: (71, "Direction_1"),
    #     110: (164, "Direction_1"),
    #     311: (66, "Direction_1"),
    # }

    def node_demand_from_data(origin_node, params=None, _data=data, _map=node_sensor_map):
        sim_steps = 1428  # must match sim_params.yaml
        sum_directions = False
        if origin_node not in _map:
            # Fallback: try to find sensor from sensor_node_df
            match = sensor_node_df[sensor_node_df["node_id"] == origin_node]
            if len(match) > 0:
                sensor_id = match["sensor_id"].values[0]
                direction = "Direction_1"
                # sum_directions = True
            else:
                # No sensor data: return zero demand
                return np.zeros(sim_steps)
        else:
            sensor_id, direction = _map[origin_node]
            if direction == None:
                sum_directions = True

        sensor_data = _data[_data["Location_ID"] == sensor_id]
        if len(sensor_data) == 0:
            return np.zeros(sim_steps)

        if sum_directions:
            demand = expand_to_10sec(sensor_data["Direction_1"] + sensor_data["Direction_2"])
        else:
            demand = expand_to_10sec(sensor_data[direction])
            if np.sum(demand) == 0:
                demand = expand_to_10sec(sensor_data["Direction_1"] + sensor_data["Direction_2"])
        demand = np.ceil(demand.values)

        # Pad or truncate to match simulation length
        if len(demand) < sim_steps:
            demand = np.pad(demand, (0, sim_steps - len(demand)))
        else:
            demand = demand[:sim_steps]

        return demand

    return node_demand_from_data


if __name__ == "__main__":
    # Load data from the new _val dataset
    data = pd.read_csv("../data/melbourne_val/melbourne.csv") # Pedestrian counts file remains the same path unless it was also copied, usually same dataset
    sensor_node_df = pd.read_csv("../data/melbourne_val/sensor_node_df.csv")

    # Create demand function and network
    demand_function = create_demand_function(data, sensor_node_df)
    env_generator = NetworkEnvGenerator()

    # Load network data (this reads adj_matrix, edge_distances, node_positions
    # and also sets self.config from sim_params.yaml)
    # env_generator.load_network_data('../data/melbourne')
    #
    # # Override config with validation config (different OD nodes and demand patterns)
    # validation_config = load_config("../data/melbourne/sim_params.yaml")
    # env_generator.config = validation_config

    network_env = env_generator.create_network(
        data_path='../data/melbourne_val',
        custom_demand_functions=[demand_function]
    )
    np.random.seed(42)

    # Run simulation
    import time
    start_time = time.time()
    sim_steps = env_generator.config['params']['simulation_steps']
    for t in range(1, sim_steps):
        network_env.network_loading(t)
        if t % 200 == 0:
            print(f"Step {t}/{sim_steps}")
    end_time = time.time()
    print(f"Simulation time: {end_time - start_time:.2f}s")
    #
    # # Save results
    output_dir = os.path.join("..", "outputs")
    output_handler = OutputHandler(base_dir=output_dir, simulation_dir="melbourne_validation")
    output_handler.save_network_state(network_env)
    output_handler.save_time_series(network_env)
    print(f"Results saved to {os.path.join(output_dir, 'melbourne_validation')}")

    # visualize
    # import json
    # import matplotlib.pyplot as plt
    # import matplotlib
    # from pednstream.utils.visualizer import NetworkVisualizer
    # with open("../data/melbourne_val/node_positions.json", 'r') as f:
    #     pos = {str(k): np.array(v) for k, v in json.load(f).items()}
    # # Create animation
    # matplotlib.use('macosx')
    # visualizer = NetworkVisualizer(simulation_dir=os.path.join(output_dir, "melbourne_validation"), pos=pos)
    # anim = visualizer.animate_network(start_time=0,
    #                                 end_time=env_generator.config['params']['simulation_steps'],
    #                                 # interval=1,
    #                                 figsize=(14, 12),
    #                                 edge_property='density')
    #
    # plt.show()
