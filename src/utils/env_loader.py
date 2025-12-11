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
        self.network = None
        self.network_data = None
        self.config = None

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
                       link_params_overrides: dict = None,
                       demand_params_overrides: dict = None):
        """Create network from saved data, simulation_params is the config dict of the yaml file"""
        if self.network_data is None:
            self.network_data = self.load_network_data(yaml_file_path)

        # Set up the simulation params, Add link-specific parameters using edge distances
        default_link_params = self.config['params']['default_link']
        
        # Apply link-specific overrides
        if link_params_overrides:
            if 'links' not in self.config['params']:
                self.config['params']['links'] = {}
            
            for link_id, params in link_params_overrides.items():
                if link_id not in self.config['params']['links']:
                    self.config['params']['links'][link_id] = {}
                self.config['params']['links'][link_id].update(params)

        if od_flows: # override the od flows
            self.config['od_flows'] = od_flows
            
        if demand_params_overrides: # override the demand params for origin nodes
             # Ensure 'demand' dictionary exists in params
            if 'demand' not in self.config['params']:
                self.config['params']['demand'] = {}
            # Update each origin's configuration
            for origin_key, params in demand_params_overrides.items():
                if origin_key not in self.config['params']['demand']:
                    self.config['params']['demand'][origin_key] = {}
                self.config['params']['demand'][origin_key].update(params)
        
        # if od_nodes_overrides: # override origin and destination nodes
        #     if 'origin_nodes' in od_nodes_overrides:
        #         self.config['origin_nodes'] = od_nodes_overrides['origin_nodes']
        #     if 'destination_nodes' in od_nodes_overrides:
        #         self.config['destination_nodes'] = od_nodes_overrides['destination_nodes']

        # Ensure 'links' dictionary exists in params
        if 'links' not in self.config['params']:
            self.config['params']['links'] = {}

        if self.network_data['edge_distances']:
            for (u, v), distance in self.network_data['edge_distances'].items():
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
        self.network = Network(
            adjacency_matrix=self.network_data['adjacency_matrix'],
            params=self.config['params'],
            origin_nodes=self.config.get('origin_nodes', []),
            destination_nodes=self.config.get('destination_nodes', []),
            # demand_pattern=self.config.get('demand_pattern', None),
            demand_pattern=custom_demand_functions,
            od_flows=self.config.get('od_flows', None),
            pos=self.network_data.get('node_positions')
        )

        return self.network

    def randomize_network(self, yaml_file_path: str, seed: int = None, randomize_params: dict = None):
        """
        Randomize network parameters
        Args:
            yaml_file_path: Path to the network data folder
            seed: Random seed for reproducibility
            randomize_params: Dictionary containing randomization parameters
        """
        reset_od_nodes = self.generate_random_od_nodes(seed)
        reset_link_params = self.generate_random_link_params(seed)
        reset_od_flows = self.generate_random_od_flows(seed)
        reset_demand_params = self.generate_random_demand_params(seed)

        
        # Create network with overrides
        network = self.create_network(
            yaml_file_path, 
            od_flows=reset_od_flows, 
            link_params_overrides=reset_link_params,
            demand_params_overrides=reset_demand_params,
        )
        return network

    def generate_random_demand_params(self, seed: int = None) -> dict:
        """
        Generate randomized demand generation parameters (patterns, lambdas).
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary {origin_key: {param: value}}
        """
        if seed is not None:
            np.random.seed(seed)
            
        origin_nodes = self.config.get('origin_nodes', [])
        demand_params = {}
        
        available_patterns = ['gaussian_peaks', 'constant', 'sudden_demand']
        
        for origin in origin_nodes:
            origin_key = f'origin_{origin}'
            
            # Randomize pattern
            pattern = np.random.choice(available_patterns)
            
            # Randomize flow parameters
            base_lambda = np.random.uniform(2.0, 10.0)
            peak_lambda = np.random.uniform(10.0, 30.0)
            
            # Ensure peak is significantly higher than base for interesting scenarios
            if peak_lambda < base_lambda + 5:
                peak_lambda = base_lambda + 5
            
            demand_params[origin_key] = {
                'pattern': pattern,
                'base_lambda': float(base_lambda),
                'peak_lambda': float(peak_lambda),
                'seed': seed # Propagate seed to demand generator
            }
            
        return demand_params

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
        
    def generate_random_od_nodes(self, seed: int = None) -> dict:
        """
        Perturbate origin and destination nodes based on default configuration.
        Maintains spatial realism by adding nodes near existing ODs.
        
        Constraint: Controller nodes cannot be origins or destinations.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary {'origin_nodes': [...], 'destination_nodes': [...]}
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Start from predefined ODs
        original_origins = self.config.get('origin_nodes', []).copy()
        original_destinations = self.config.get('destination_nodes', []).copy()
        
        # Get adjacency matrix for neighbor lookup
        adj_matrix = self.network_data['adjacency_matrix']
        
        def get_neighbors(node_list, hop=1):
            """Get k-hop neighbors of nodes in node_list"""
            neighbors = set()
            for node in node_list:
                # For symmetric adjacency matrix, only need to check one direction
                neighbors.update(np.where(adj_matrix[node, :] == 1)[0].tolist())
            
            if hop == 2:
                # 2-hop neighbors
                hop2_neighbors = set()
                for n in neighbors:
                    hop2_neighbors.update(np.where(adj_matrix[n, :] == 1)[0].tolist())
                neighbors.update(hop2_neighbors)
            
            return list(neighbors)
        
        # === Perturbate Origins ===
        new_origins = original_origins.copy()
        
        # 1. ADD: Add new origins near existing origins (spatial clustering)
        if np.random.random() < 0.5:
            neighbor_candidates = get_neighbors(new_origins, hop=2)
            # Exclude controller nodes and existing origins
            candidates = [n for n in neighbor_candidates 
                         if n not in new_origins and n not in self.network.controller_nodes]
            if candidates:
                num_to_add = np.random.randint(1, min(2, len(candidates) + 1))
                new_origins.extend([int(x) for x in np.random.choice(candidates, num_to_add, replace=False)])
        
        # 2. REMOVE: Randomly remove origins - keep at least 1
        if len(new_origins) > 1 and np.random.random() < 0.5:
            num_to_remove = np.random.randint(1, min(2, len(new_origins)))
            indices_to_remove = np.random.choice(len(new_origins), num_to_remove, replace=False)
            new_origins = [o for i, o in enumerate(new_origins) if i not in indices_to_remove]
        
        # 3. SWAP: Replace an origin with a spatial neighbor
        if np.random.random() < 0.5:
            origin_to_swap = np.random.choice(new_origins)
            neighbors = get_neighbors([origin_to_swap], hop=2)
            # Exclude controller nodes and existing origins
            valid_neighbors = [n for n in neighbors 
                              if n not in new_origins and n not in self.network.controller_nodes]
            if valid_neighbors:
                new_node = int(np.random.choice(valid_neighbors))
                new_origins[new_origins.index(origin_to_swap)] = new_node
        
        # === Perturbate Destinations ===
        new_destinations = original_destinations.copy()
        
        # ADD: Add new destinations near existing destinations
        if np.random.random() < 0.5:
            neighbor_candidates = get_neighbors(new_destinations, hop=2)
            # Exclude controller nodes and existing destinations
            candidates = [n for n in neighbor_candidates 
                         if n not in new_destinations and n not in self.network.controller_nodes]
            if candidates:
                num_to_add = np.random.randint(1, min(3, len(candidates) + 1))
                new_destinations.extend([int(x) for x in np.random.choice(candidates, num_to_add, replace=False)])
        
        # REMOVE: Remove destinations (keep at least as many as origins)
        if len(new_destinations) > len(new_origins) and np.random.random() < 0.5:
            removable = [d for d in new_destinations if d not in new_origins]
            if removable:
                num_to_remove = np.random.randint(1, min(2, len(removable) + 1))
                to_remove = [int(x) for x in np.random.choice(removable, num_to_remove, replace=False)]
                new_destinations = [d for d in new_destinations if d not in to_remove]
        
        # Ensure all values are native Python int (not np.int64)
        new_origins = [int(x) for x in new_origins]
        new_destinations = [int(x) for x in new_destinations]
        
        # # ENFORCE CONSTRAINT: All origins must be in destinations
        # new_destinations = list(set(new_destinations) | set(new_origins))
        self.config['origin_nodes'] = new_origins
        self.config['destination_nodes'] = new_destinations
        return {'origin_nodes': new_origins, 'destination_nodes': new_destinations}
        
        

    def generate_random_link_params(self, seed: int = None) -> dict:
        """
        Generate randomized parameters for specific links.
        This focuses on local perturbations (incidents, bottlenecks) rather than global shifts.
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Get all valid links from the dataset
        # We access the data directly to know the topology
        if not self.network_data:
            self.network_data = self.load_network_data(self.data_dir.name if self.data_dir.name != 'data' else 'delft')
        
        if not self.config:
            self.config = self.load_config(self.data_dir.name if self.data_dir.name != 'data' else 'delft')
        
        valid_links = []
        if 'edge_distances' in self.network_data and self.network_data['edge_distances']:
             # Use only unique corridors (assuming bidirectional symmetry)
             valid_links = [f"{u}_{v}" for (u, v) in self.network_data['edge_distances'].keys() if u < v]
        elif 'adjacency_matrix' in self.network_data:
             adj_matrix = self.network_data['adjacency_matrix']
             # Find all pairs (u, v) where adjacency_matrix[u, v] == 1 and u < v (upper triangle only)
             rows, cols = np.where(adj_matrix == 1)
             valid_links = [f"{u}_{v}" for u, v in zip(rows, cols) if u < v]
        
        defaults = self.config['params']['default_link']
        link_overrides = {}
        
        # Select a subset of links to perturb (e.g., 20%)
        if valid_links:
            num_links_to_change = int(len(valid_links) * 0.2)
            if num_links_to_change > 0:
                target_links = np.random.choice(valid_links, num_links_to_change, replace=False)
                
                for link_id in target_links:
                    params = {}
                    
                    # Scenario A: Capacity change (bottleneck) -> Affects k_critical and k_jam
                    if np.random.random() < 0.5:
                        # Reduce capacity by 20-50%
                        factor = np.random.uniform(0.6, 1.2)
                        
                        current_k_crit = self.config['params']['links'].get(link_id, {}).get('k_critical', defaults['k_critical'])
                        current_k_jam = self.config['params']['links'].get(link_id, {}).get('k_jam', defaults['k_jam'])
                        
                        params['k_critical'] = max(0.5, current_k_crit * factor)
                        params['k_jam'] = max(params['k_critical'] * 2.0, current_k_jam * factor)
                        
                    # Scenario B: Speed reduction (wet floor / congestion)
                    if np.random.random() < 0.5:
                        current_ffs = self.config['params']['links'].get(link_id, {}).get('free_flow_speed', defaults['free_flow_speed'])
                        params['free_flow_speed'] = current_ffs * np.random.uniform(0.6, 0.9)
                    
                    if params:
                        link_overrides[link_id] = params
                        # # Explicitly set reverse link to ensure symmetry regardless of iteration order in create_network
                        # u, v = link_id.split('_')
                        # reverse_id = f"{v}_{u}"
                        # link_overrides[reverse_id] = params.copy()

        return link_overrides

if __name__ == "__main__":
    data_manager = NetworkEnvGenerator()
    network = data_manager.create_network("delft")
    # Run simulation
    for t in range(1, data_manager.config['params']['simulation_steps']):
        network.network_loading(t)