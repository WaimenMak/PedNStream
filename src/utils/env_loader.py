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
from typing import List, Callable

class NetworkEnvGenerator:
    """The input of this class is the simulation parameters, and the output is the network environment."""

    def __init__(self, data_dir="data"):
        # self.simulation_params = simulation_params
        # create the data directory 'project_root/data'
        project_root = Path(__file__).resolve().parent.parent.parent
        self.data_dir = project_root / data_dir
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

        #if exists, load the edge distances
        if os.path.exists(os.path.join(self.data_dir, f"{data_path}", "edge_distances.pkl")):
            with open(os.path.join(self.data_dir, f"{data_path}", "edge_distances.pkl"), 'rb') as f:
                edge_distances = pickle.load(f)
        else:
            edge_distances = None

        #if adj not in yaml, load the adjacency matrix
        if 'adjacency_matrix' not in self.config:
            adjacency_matrix = np.load(os.path.join(self.data_dir, f"{data_path}", "adj_matrix.npy"))
        else:
            adjacency_matrix = self.config['adjacency_matrix']

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

    def create_network(self, yaml_file_path: str,
                       custom_demand_functions: List[Callable] = None,
                       od_flows: dict = None,
                       link_params_overrides: dict = None):
        """Create network from saved data, simulation_params is the config dict of the yaml file"""
        network_data = self.load_network_data(yaml_file_path)

        # Set up the simulation params, Add link-specific parameters using edge distances
        default_link_params = self.config['params']['default_link']
        if link_params_overrides: # override the default link parameters
            default_link_params.update(link_params_overrides)

        if od_flows: # override the od flows
            self.config['od_flows'] = od_flows
        # Ensure 'links' dictionary exists in params
        if 'links' not in self.config['params']:
            self.config['params']['links'] = {}

        if network_data['edge_distances']:
            for (u, v), distance in network_data['edge_distances'].items():
                link_id = f"{u}_{v}"

                # Get existing link-specific params, or an empty dict if none
                link_specific_params = self.config['params']['links'].get(link_id, {})

                # Build the parameters for the link, starting with defaults and overriding
                final_params = default_link_params.copy()
                final_params.update(link_specific_params)

                # Explicitly set length from distance data and ensure width uses the default
                final_params['length'] = distance
                # final_params['width'] = default_link_params['width']

                self.config['params']['links'][link_id] = final_params
                # if reverse link not in the config, add it
                if f"{v}_{u}" not in self.config['params']['links']:
                    self.config['params']['links'][f"{v}_{u}"] = final_params

        # Create network
        network = Network(
            adjacency_matrix=network_data['adjacency_matrix'],
            params=self.config['params'],
            origin_nodes=self.config.get('origin_nodes', []),
            destination_nodes=self.config.get('destination_nodes', []),
            # demand_pattern=self.config.get('demand_pattern', None),
            demand_pattern=custom_demand_functions,
            od_flows=self.config.get('od_flows', None),
            pos=network_data.get('node_positions')
        )

        return network

    def randomize_network(self, yaml_file_path: str, seed: int = None, randomize_params: dict = None):
        """
        Randomize network parameters
        Args:
            yaml_file_path: Path to the network data folder
            seed: Random seed for reproducibility
            randomize_params: Dictionary containing randomization parameters
        """
        reset_link_params = self.generate_random_link_params(seed)
        reset_od_flows = self.generate_random_od_flows(seed)
        
        # Create network with overrides
        network = self.create_network(yaml_file_path, od_flows=reset_od_flows, link_params_overrides=reset_link_params)
        return network

    def generate_random_od_flows(self, seed: int = None) -> dict:
        """
        Generate randomized OD flows ratio for the network.
        The values represent the relative weight/preference for each destination, not absolute flow.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary {(o,d): weight_array} where weight_array is numpy array of size simulation_steps+1
        """
        if seed is not None:
            np.random.seed(seed)
            
        origin_nodes = self.config.get('origin_nodes', [])
        destination_nodes = self.config.get('destination_nodes', [])
        simulation_steps = self.config['params']['simulation_steps']
        
        od_flows = {}
        
        for o in origin_nodes:
            for d in destination_nodes:
                if o == d:
                    continue
                
                # Assign a random weight to this OD pair (e.g., uniform between 1 and 10)
                # This controls the split ratio: flow(o->d) / flow(o->d') = weight(d) / weight(d')
                base_weight = np.random.uniform(1.0, 10.0)
                
                # Create weight array (constant over time for this episode)
                # This ensures consistent destination preference throughout the simulation
                weights = np.full(simulation_steps + 1, base_weight)
                
                od_flows[(o, d)] = weights
                
        return od_flows
        
        

    def generate_random_link_params(self, seed: int = None) -> dict:
        """Generate randomized link parameters based on defaults."""
        if seed is not None:
            np.random.seed(seed)
            
        # Get base defaults (assuming config is loaded)
        defaults = self.config['params']['default_link']
        random_params = defaults.copy()
        
        # 1. Randomized Fundamental Diagram Parameters
        # Free flow speed: +/- 10%
        random_params['free_flow_speed'] = np.random.uniform(
            defaults['free_flow_speed'] * 0.9, 
            defaults['free_flow_speed'] * 1.1
        )
        
        # Critical density: +/- 10%
        random_params['k_critical'] = np.random.uniform(
            defaults['k_critical'] * 0.9, 
            defaults['k_critical'] * 1.1
        )
        
        # Jam density: +/- 10% (ensure it stays > k_critical)
        random_params['k_jam'] = np.random.uniform(
            defaults['k_jam'] * 0.9, 
            defaults['k_jam'] * 1.1
        )
        # Safety check
        if random_params['k_jam'] <= random_params['k_critical']:
            random_params['k_jam'] = random_params['k_critical'] * 2.0

        # 2. Randomized Behavior/Noise
        # Speed noise standard deviation (0.0 to 0.1)
        random_params['speed_noise_std'] = np.random.uniform(0.0, 0.1)
        
        # # Bi-directional factor (0.8 to 1.2 for sensitivity analysis)
        # if 'bi_factor' in defaults:
        #     random_params['bi_factor'] = np.random.uniform(0.8, 1.2)

        return random_params

if __name__ == "__main__":
    data_manager = NetworkEnvGenerator()
    network = data_manager.create_network("delft")
    # Run simulation
    for t in range(1, data_manager.config['params']['simulation_steps']):
        network.network_loading(t)