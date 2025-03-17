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

    def create_network(self, yaml_file_path: str):
        """Create network from saved data, simulation_params is the config dict of the yaml file"""
        network_data = self.load_network_data(yaml_file_path)

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

    def randomize_network(self, network, randomize_params):
        """
        Randomize network parameters
        Args:
            network: Network object
            randomize_params: Dictionary containing randomization parameters
        """
        # Implement randomization logic here
        pass

if __name__ == "__main__":
    data_manager = NetworkEnvGenerator()
    network = data_manager.create_network("delft")
    # Run simulation
    for t in range(1, data_manager.config['params']['simulation_steps']):
        network.network_loading(t)