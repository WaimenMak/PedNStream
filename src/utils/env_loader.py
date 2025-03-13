# -*- coding: utf-8 -*-
# @Time    : 11/03/2025 17:43
# @Author  : mmai
# @FileName: load_network_data
# @Software: PyCharm

"""
This file is used to generate different network environments for the RL environment.
The NetworkEnvGenerator class is used to load the network data and generate the network environment.
"""

import json
from src.LTM.network import Network
import os
from pathlib import Path
import numpy as np
import pickle
from src.utils.config import load_config

class NetworkEnvGenerator:
    """The input of this class is the simulation parameters, and the output is the network environment."""

    def __init__(self, data_dir="data"):
        # self.simulation_params = simulation_params

        self.data_dir = Path(os.path.join("..", data_dir))
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # def save_network_data(self, simulation_params: dict, data: dict):
    #     """
    #     Save network data to file

    #     Args:
    #         simulation_params: Dictionary containing simulation parameters
    #         data: Dictionary containing network data including:
    #             - adjacency_matrix
    #             - edge_distances
    #             - node_positions
    #             - other network attributes
    #     """
    #     file_path = self.data_dir / f"{simulation_params['network_name']}.json"

    #     # Convert numpy arrays to lists if present
    #     save_data = {}
    #     for key, value in data.items():
    #         if isinstance(value, np.ndarray):
    #             save_data[key] = value.tolist()
    #         else:
    #             save_data[key] = value

    #     with open(file_path, 'w') as f:
    #         json.dump(save_data, f, indent=2)

    def load_network_data(self, data_path: str) -> dict:
        """
        Load network data from file

        Args:
            data_path: Path to the network data folder

        Returns:
            Dictionary containing network data
        """
        yaml_file_path = os.path.join(self.data_dir, f"{data_path}", "sim_params.yaml")

        if not os.path.exists(yaml_file_path):
            raise FileNotFoundError(f"Network data file not found: {yaml_file_path}")

        # load the simulation parameters
        self.config = load_config(yaml_file_path)

        # load the edge distances
        with open(os.path.join(self.data_dir, f"{data_path}", "edge_distances.pkl"), 'rb') as f:
            edge_distances = pickle.load(f)

        # load the adjacency matrix
        adjacency_matrix = np.load(os.path.join(self.data_dir, f"{data_path}", "adj_matrix.npy"))

        # load the node positions if it exists
        if os.path.exists(os.path.join(self.data_dir, f"{data_path}", "node_positions.json")):
            with open(os.path.join(self.data_dir, f"{data_path}", "node_positions.json"), 'r') as f:
                node_positions = {str(node): pos for node, pos in json.load(f).items()}
        else:
            node_positions = None

        # aggregate the data
        data = {
            "adjacency_matrix": adjacency_matrix,
            "edge_distances": edge_distances,
            "node_positions": node_positions
        }

        return data

# Usage example:
# def save_network_configuration(G, simulation_params: dict):
#     """Save network configuration from NetworkX graph"""
#     data_manager = NetworkEnvGenerator()

#     network_data = {
#         'adjacency_matrix': nx.to_numpy_array(G),
#         'edge_distances': dict(nx.get_edge_attributes(G, 'length')),
#         'node_positions': dict(nx.get_node_attributes(G, 'pos')),
#         'network_params': {
#             'default_link': {
#                 'width': 1,
#                 'free_flow_speed': 1.5,
#                 'k_critical': 2,
#                 'k_jam': 10
#             }
#         }
#     }

#     data_manager.save_network_data(simulation_params, network_data)

    def create_network(self, yaml_file_path: str):
        """Create network from saved data, simulation_params is the config dict of the yaml file"""
        network_data = self.load_network_data(yaml_file_path)

        # Merge simulation params with network params
        # params = {
        #     **network_data['network_params'], # include adjacency matrix, edge distances, node positions
        #     **self.config, # include simulation time, unit time, and default link parameters
        # }

        # Add link-specific parameters using edge distances
        for (u, v), distance in network_data['edge_distances'].items():
            link_id = f"{u}_{v}"
            self.config['params']['links'][link_id] = {
                'length': distance,
                'width': self.config['params']['default_link']['width'],
                'free_flow_speed': self.config['params']['default_link']['free_flow_speed'],
                'k_critical': self.config['params']['default_link']['k_critical'],
                'k_jam': self.config['params']['default_link']['k_jam'],
            }

        # Create network
        network = Network(
            adjacency_matrix=network_data['adjacency_matrix'],
            params=self.config['params'],
            origin_nodes=self.config.get('origin_nodes', []),
            destination_nodes=self.config.get('destination_nodes', []),
            demand_pattern=self.config.get('demand_pattern', None),
            od_flows=self.config.get('od_flows', None),
            pos=network_data.get('node_positions')
        )

        return network

if __name__ == "__main__":
    data_manager = NetworkEnvGenerator()
    network = data_manager.create_network("delft")
    # Run simulation
    for t in range(1, data_manager.config['params']['simulation_steps']):
        network.network_loading(t)