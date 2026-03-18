"""Code to generate different network environments for the RL environment."""

# -*- coding: utf-8 -*-
# @Time    : 11/03/2025 17:43
# @Author  : mmai
# @FileName: load_network_data
# @Software: PyCharm

import json
import numpy as np
import pickle
import copy
from pednstream.ltm.network import Network
from pathlib import Path
from pednstream.utils.config import load_config
from typing import List, Callable, Dict


class NetworkEnvGenerator:
    """Represents a network environment."""

    # to load the network data and generate the network environment.
    # The input of this class is the simulation parameters, and the output is the network environment."""

    def __init__(self, data_dir: str = "data/"):
        """Creates network environment using simulation data files.

        Args:
            data_dir (str): path to directory containing data files
        """
        # Resolve input data root from working directory by default.
        # If an absolute path is passed, use it as-is.
        self.data_path = self._normalize_path(data_dir)
        self.network = None
        self.network_data = None
        self.config = {}
        self._original_config = (
            None  # Store pristine config from YAML for randomization base
        )

    def _normalize_path(self, directory_path: str):
        """Normalize a directory path. If path is relative, the path to current directory used in the normalization.

        Args:
            directory_path (str): a path to a directory

        Returns:
            Normalized path, an absolute path
        """
        _path = Path(directory_path).expanduser()
        if not _path.is_absolute():
            current_directory = Path.cwd()
            _path = current_directory / _path
        return _path.resolve()

    def load_network_data(self, data_dir: str = "") -> dict:
        """Loads network data from file.

        Args:
            data_dir (str): Path to data directory. Enables to load data from a different directory after initialization.

        Returns:
            Dictionary containing network aggregated data
        """
        data_path = self.data_path
        if data_dir:
            # overwrites initial data_path for this method
            data_path = self._normalize_path(data_dir)

        yaml_file_path = data_path / "sim_params.yaml"
        if not yaml_file_path.exists():
            raise FileNotFoundError(f"'sim_params.yaml' not found in: {str(data_path)}")

        # load the simulation parameters
        self.config = load_config(str(yaml_file_path))
        # Store original config for randomization base (only on first load)
        if self._original_config is None:
            self._original_config = copy.deepcopy(self.config)

        # load edge distances if present in data path
        edge_distances_path = data_path / "edge_distances.pkl"
        if edge_distances_path.exists():
            with open(edge_distances_path, "rb") as f:
                edge_distances = pickle.load(f)
            if not isinstance(edge_distances, Dict):
                raise TypeError(
                    f"Edge Distances. Pickled file must contain a dictionary, got: {type(edge_distances)}"
                )
        else:
            edge_distances = {}

        # load the adjacency matrix form configuration or from file
        if "adjacency_matrix" not in self.config:
            adjacency_matrix = np.load(data_path / "adj_matrix.npy")
        else:
            adjacency_matrix = self.config["adjacency_matrix"]

        # load the node positions if it exists
        node_positions_path = data_path / "node_positions.json"
        if node_positions_path.exists():
            with open(node_positions_path, "r") as f:
                node_positions = {str(node): pos for node, pos in json.load(f).items()}
        else:
            node_positions = {}

        # aggregate the data
        data = {
            "adjacency_matrix": adjacency_matrix,
            "edge_distances": edge_distances,
            "node_positions": node_positions,
        }

        return data

    def create_network(
        self,
        data_path: str,
        custom_demand_functions: List[Callable] = [],
        od_flows: dict = {},
        link_params_overrides: dict = {},
        demand_params_overrides: dict = {},
        verbose: bool = True,
    ):
        """Create network from saved data, simulation_params is the config dict of the yaml file

        Args:
            data_path: Name of the folder in the data directory (e.g., 'butterfly_scC', 'delft')
            verbose: If True, enable logging output. Default True for backward compatibility.
        """
        if self.network_data is None:
            self.network_data = self.load_network_data(data_path)

        # Set up the simulation params, Add link-specific parameters using edge distances
        default_link_params = self.config["params"]["default_link"]

        # Apply link-specific overrides for randomization
        if link_params_overrides:
            if "links" not in self.config["params"]:
                self.config["params"]["links"] = {}

            for link_id, params in link_params_overrides.items():
                if link_id not in self.config["params"]["links"]:
                    self.config["params"]["links"][link_id] = {}
                self.config["params"]["links"][link_id].update(params)

        if od_flows:  # override the od flows
            self.config["od_flows"] = od_flows

        if (
            demand_params_overrides
        ):  # override the demand params for origin nodes for randomization
            # Ensure 'demand' dictionary exists in params
            if "demand" not in self.config["params"]:
                self.config["params"]["demand"] = {}
            # Update each origin's configuration
            for origin_key, params in demand_params_overrides.items():
                if origin_key not in self.config["params"]["demand"]:
                    self.config["params"]["demand"][origin_key] = {}
                self.config["params"]["demand"][origin_key].update(params)

        # Ensure 'links' dictionary exists in params
        if "links" not in self.config["params"]:
            self.config["params"]["links"] = {}

        if self.network_data["edge_distances"]:
            for (u, v), distance in self.network_data["edge_distances"].items():
                link_id = f"{u}_{v}"

                # Get existing link-specific params, or an empty dict if none
                link_specific_params = self.config["params"]["links"].get(link_id, {})

                # Build the parameters for the link, starting with defaults and overriding
                final_params = default_link_params.copy()
                final_params.update(link_specific_params)

                # Explicitly set length from distance data and ensure width uses the default
                final_params["length"] = distance
                # final_params['width'] = default_link_params['width']

                self.config["params"]["links"][link_id] = final_params
                # if reverse link not in the config, add it
                if f"{v}_{u}" not in self.config["params"]["links"]:
                    # Create a copy of final_params for reverse link and swap front/back gate widths
                    reverse_params = final_params.copy()
                    # TODO: Do we need the code below at all?
                    # original_front = reverse_params.pop('front_gate_width', None)
                    # original_back = reverse_params.pop('back_gate_width', None)
                    # if original_front is not None:
                    #     reverse_params['back_gate_width'] = original_front
                    # if original_back is not None:
                    #     reverse_params['front_gate_width'] = original_back
                    self.config["params"]["links"][f"{v}_{u}"] = reverse_params

        # Create network
        self.network = Network(
            adjacency_matrix=self.network_data["adjacency_matrix"],
            params=self.config["params"],
            origin_nodes=self.config.get("origin_nodes", []),
            destination_nodes=self.config.get("destination_nodes", []),
            # demand_pattern=self.config.get('demand_pattern', None),
            demand_pattern=custom_demand_functions,
            od_flows=self.config.get("od_flows", None),
            pos=self.network_data.get("node_positions"),
            verbose=verbose,
        )

        return self.network

    def randomize_network(
        self,
        data_path: str,
        seed: int = None,
        randomize_params: dict = None,
        verbose: bool = True,
    ):
        """
        Randomize network parameters
        Args:
            data_path: Name of the folder in the data directory (e.g., 'butterfly_scC', 'delft')
            seed: Random seed for reproducibility, not used in RL
            randomize_params: Dictionary containing randomization parameters
            verbose: If True, enable logging output. Default True for backward compatibility.
        """
        # Set random seed for reproducibility BEFORE any random operations
        if seed is not None:
            np.random.seed(seed)
            import random

            random.seed(seed)

        # reset_od_nodes = self.generate_random_od_nodes()
        # reset_link_params = self.generate_random_link_params()
        reset_od_flows = self.generate_random_od_flows()
        reset_demand_params = self.generate_random_demand_params()
        # reset_gate_widths = self.generate_random_gate_widths()

        # Create network with overrides
        network = self.create_network(
            data_path,
            od_flows=reset_od_flows,
            # link_params_overrides=reset_link_params,
            # link_params_overrides=reset_gate_widths,
            demand_params_overrides=reset_demand_params,
            verbose=verbose,
        )
        return network

    # COTINUE HERE:
    def generate_random_demand_params(self) -> dict:
        """
        Generate randomized demand  parameters (patterns, lambdas).
        Uses original YAML config values as base and applies perturbation.
        For new origins not in original config, randomly picks params from an existing origin.

        Args:

        Returns:
            Dictionary {origin_key: {param: value}}
        """
        # Use randomized origin nodes, but original YAML demand config as perturbation base
        origin_nodes = self.config.get("origin_nodes", [])
        original_demand_config = self._original_config["params"].get("demand", {})
        demand_params = {}

        available_patterns = [
            "gaussian_peaks",
            "constant",
            "sudden_demand",
            "multi_peaks",
            "single_peak",
        ]

        # Default values if not specified in config
        default_base_lambda = 10.0
        default_peak_lambda = 30.0

        # Collect all original demand configs for fallback
        original_origin_keys = list(original_demand_config.keys())

        for origin in origin_nodes:
            origin_key = f"origin_{origin}"

            # Try to get config for this origin from original YAML
            if origin_key in original_demand_config:
                # Origin exists in original config - use its params
                origin_config = original_demand_config[origin_key]
            elif original_origin_keys:
                # New origin not in original config - pick params from a random original origin
                random_original_key = np.random.choice(original_origin_keys)
                origin_config = original_demand_config[random_original_key]
            else:
                # No original config at all - use defaults
                origin_config = {}

            # Randomize pattern (use existing as more likely choice)
            existing_pattern = origin_config.get("pattern", None)
            if existing_pattern and np.random.random() < 0.5:
                # 50% chance to keep existing pattern
                pattern = existing_pattern
            else:
                pattern = np.random.choice(available_patterns)

            # Get base values from config or use defaults
            config_base_lambda = origin_config.get("base_lambda", default_base_lambda)
            config_peak_lambda = origin_config.get("peak_lambda", default_peak_lambda)

            # Apply perturbation: +/- 30% of the config value
            perturbation_factor = np.random.uniform(0.7, 1.5)
            base_lambda = config_base_lambda * perturbation_factor

            perturbation_factor = np.random.uniform(0.7, 1.5)
            peak_lambda = config_peak_lambda * perturbation_factor

            # Ensure peak is higher than base
            if peak_lambda < base_lambda + 5:
                peak_lambda = base_lambda + 5

            demand_params[origin_key] = {
                "pattern": pattern,
                "base_lambda": float(base_lambda),
                "peak_lambda": float(peak_lambda),
            }

            # For single_peak pattern, randomize peak_position (0.1-0.9)
            if pattern == "single_peak":
                demand_params[origin_key]["peak_position"] = float(
                    np.random.uniform(0.1, 0.9)
                )

        # Optionally shuffle demand profiles across origins for extra diversity
        demand_params = self._shuffle_demand_among_origins(
            demand_params, shuffle_prob=0.5
        )

        return demand_params

    def _shuffle_demand_among_origins(
        self, demand_params: dict, shuffle_prob: float = 0.3
    ) -> dict:
        """
        Shuffle demand configurations among origins with some probability.
        When triggered, randomly permutes the demand profiles (pattern + lambdas)
        across origin nodes, so that e.g. a typically low-demand origin may receive
        a high-demand profile and vice versa.

        This increases training diversity by decoupling demand characteristics from
        their original spatial assignment.

        Args:
            demand_params: Dictionary {origin_key: {pattern, base_lambda, peak_lambda}}
                           as produced by generate_random_demand_params.
            shuffle_prob: Probability of performing the shuffle (default 0.3).

        Returns:
            demand_params with profiles potentially reassigned across origins.
        """
        origin_keys = list(demand_params.keys())
        if len(origin_keys) < 2 or np.random.random() >= shuffle_prob:
            return demand_params

        # Extract the profiles (values) and shuffle them
        profiles = [demand_params[k] for k in origin_keys]
        shuffled_indices = np.random.permutation(len(origin_keys))

        shuffled_params = {}
        for i, key in enumerate(origin_keys):
            shuffled_params[key] = profiles[shuffled_indices[i]]

        return shuffled_params

    def generate_random_od_flows(self) -> dict:
        """
        Generate randomized OD flows ratio for the network.
        The values represent the relative weight/preference for each destination, not absolute flow.

        Args:

        Returns:
            Dictionary {(o,d): weight_array} where weight_array is numpy array of size simulation_steps+1
        """
        origin_nodes = self.config.get("origin_nodes", [])
        destination_nodes = self.config.get("destination_nodes", [])
        simulation_steps = self.config["params"]["simulation_steps"]

        od_flows = {}

        for o in origin_nodes:
            for d in destination_nodes:
                if o == d:
                    continue

                # Generate time-varying weights to represent changing OD proportions
                # Choose a random pattern for this OD pair
                pattern_type = np.random.choice(
                    ["constant", "linear", "sine", "random_walk"]
                )

                if pattern_type == "constant":
                    # Constant weight over time
                    base_weight = np.random.uniform(0, 10.0)
                    weights = np.full(simulation_steps + 1, base_weight)

                elif pattern_type == "linear":
                    # Linear increase or decrease
                    start_weight = np.random.uniform(0, 10.0)
                    end_weight = np.random.uniform(1.0, 10.0)
                    weights = np.linspace(
                        start_weight, end_weight, simulation_steps + 1
                    )

                elif pattern_type == "sine":
                    # Sinusoidal variation (e.g., rush hour patterns)
                    base_weight = np.random.uniform(3.0, 7.0)
                    amplitude = np.random.uniform(1.0, 3.0)
                    frequency = np.random.uniform(0.5, 2.0)
                    time_steps = np.arange(simulation_steps + 1)
                    weights = base_weight + amplitude * np.sin(
                        2 * np.pi * frequency * time_steps / simulation_steps
                    )
                    weights = np.clip(
                        weights, 0.0, 10.0
                    )  # Keep within reasonable bounds

                else:  # random_walk
                    # Random walk for unpredictable variation
                    base_weight = np.random.uniform(3.0, 7.0)
                    weights = [base_weight]
                    for _ in range(simulation_steps):
                        step = np.random.uniform(-0.5, 0.5)
                        new_weight = np.clip(weights[-1] + step, 0.0, 10.0)
                        weights.append(new_weight)
                    weights = np.array(weights)

                od_flows[(o, d)] = weights

        return od_flows

    def generate_random_od_nodes(self) -> dict:
        """
        Add 1-2 random origin nodes to the existing configuration.
        Keeps original origins and destinations unchanged, only adds new origins.

        Constraint: Controller nodes cannot be origins or destinations.

        Args:

        Returns:
            Dictionary {'origin_nodes': [...], 'destination_nodes': [...]}
        """
        # Start from predefined ODs in original YAML config (not modified config)
        original_origins = self._original_config.get("origin_nodes", []).copy()
        original_destinations = self._original_config.get(
            "destination_nodes", []
        ).copy()

        # Get adjacency matrix for neighbor lookup
        adj_matrix = self.network_data["adjacency_matrix"]

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

        # Add 1-2 new origins: randomly choose from neighbors or from destinations
        new_origins = original_origins.copy()

        if np.random.random() < 0:
            # Option 1: Add from spatial neighbors of existing origins
            neighbor_candidates = get_neighbors(new_origins, hop=2)
            candidates = [
                n
                for n in neighbor_candidates
                if n not in new_origins
                and n not in self.network.controller_nodes
                and n not in original_destinations
            ]
        else:
            # Option 2: Add from original destination nodes
            candidates = [
                n
                for n in original_destinations
                if n not in new_origins and n not in self.network.controller_nodes
            ]

        if candidates:
            num_to_add = np.random.randint(
                1, min(3, len(candidates) + 1)
            )  # Add 1-2 nodes
            new_origins.extend(
                [
                    int(x)
                    for x in np.random.choice(candidates, num_to_add, replace=False)
                ]
            )

        # Keep destinations unchanged
        new_destinations = original_destinations.copy()

        # Ensure all values are native Python int (not np.int64)
        new_origins = [int(x) for x in new_origins]
        new_destinations = [int(x) for x in new_destinations]

        self.config["origin_nodes"] = new_origins
        self.config["destination_nodes"] = new_destinations
        return {"origin_nodes": new_origins, "destination_nodes": new_destinations}

    def generate_random_gate_widths(self, seed: int = None) -> dict:
        """
        Generate randomized initial back_gate_width for outgoing links of controller nodes (gaters).

        The back_gate_width is randomly initialized from uniform distribution [0, link_width].

        Args:
            seed: Random seed for reproducibility

        Returns:
            Dictionary {link_id: {'back_gate_width': value}}
        """
        if seed is not None:
            np.random.seed(seed)

        gate_width_overrides = {}

        # Get controller configuration
        controller_config = self._original_config.get("params", {}).get(
            "controllers", {}
        )
        if not controller_config.get("enabled", False):
            return gate_width_overrides

        controller_nodes = set(map(int, controller_config.get("nodes", [])))
        if not controller_nodes:
            return gate_width_overrides

        # Get adjacency matrix to find outgoing links
        adj_matrix = self.network_data["adjacency_matrix"]

        default_link_params = self._original_config.get("params", {}).get(
            "default_link", {}
        )
        default_width = default_link_params.get("width", 2.0)
        link_params = self._original_config.get("params", {}).get("links", {})

        # Iterate through controller nodes and randomize their outgoing links' back_gate_width
        for node in controller_nodes:
            # Find outgoing neighbors from adjacency matrix
            outgoing_neighbors = np.where(adj_matrix[node, :] == 1)[0]

            for neighbor in outgoing_neighbors:
                link_id = f"{node}_{neighbor}"
                reverse_link_id = f"{neighbor}_{node}"

                # Get link-specific width or use default
                link_specific_params = link_params.get(link_id, {})
                link_width = link_specific_params.get("width", default_width)

                # Randomize back_gate_width from uniform [0, link_width]
                back_gate = np.random.uniform(0, link_width)
                rev_back_gate = np.random.uniform(0, link_width)

                # Set back_gate_width for outgoing link
                gate_width_overrides[link_id] = {"back_gate_width": back_gate}

                # Set front_gate_width for reverse link
                if reverse_link_id not in gate_width_overrides:
                    gate_width_overrides[reverse_link_id] = {}
                gate_width_overrides[reverse_link_id]["back_gate_width"] = rev_back_gate

        return gate_width_overrides

    def generate_random_link_params(self) -> dict:
        """
        Generate randomized parameters for specific links.
        This focuses on local perturbations (incidents, bottlenecks) rather than global shifts.
        """
        # Get all valid links from the dataset
        # We access the data directly to know the topology
        if not self.network_data:
            self.network_data = self.load_network_data(
                self.data_dir.name if self.data_dir.name != "data" else "delft"
            )

        if not self.config:
            self.config = self.load_config(
                self.data_dir.name if self.data_dir.name != "data" else "delft"
            )

        valid_links = []
        if (
            "edge_distances" in self.network_data
            and self.network_data["edge_distances"]
        ):
            # Use only unique corridors (assuming bidirectional symmetry)
            valid_links = [
                f"{u}_{v}"
                for (u, v) in self.network_data["edge_distances"].keys()
                if u < v
            ]
        elif "adjacency_matrix" in self.network_data:
            adj_matrix = self.network_data["adjacency_matrix"]
            # Find all pairs (u, v) where adjacency_matrix[u, v] == 1 and u < v (upper triangle only)
            rows, cols = np.where(adj_matrix == 1)
            valid_links = [f"{u}_{v}" for u, v in zip(rows, cols) if u < v]

        defaults = self.config["params"]["default_link"]
        link_overrides = {}

        # Select a subset of links to perturb (e.g., 20%)
        if valid_links:
            num_links_to_change = int(len(valid_links) * 0.2)
            if num_links_to_change > 0:
                target_links = np.random.choice(
                    valid_links, num_links_to_change, replace=False
                )

                for link_id in target_links:
                    params = {}

                    # Scenario A: Capacity change (bottleneck) -> Affects k_critical and k_jam
                    if np.random.random() < 0.5:
                        # Reduce capacity by 20-50%
                        factor = np.random.uniform(0.6, 1.2)

                        current_k_crit = (
                            self.config["params"]["links"]
                            .get(link_id, {})
                            .get("k_critical", defaults["k_critical"])
                        )
                        current_k_jam = (
                            self.config["params"]["links"]
                            .get(link_id, {})
                            .get("k_jam", defaults["k_jam"])
                        )

                        params["k_critical"] = max(0.5, current_k_crit * factor)
                        params["k_jam"] = max(
                            params["k_critical"] * 2.0, current_k_jam * factor
                        )

                    # Scenario B: Speed reduction (wet floor / congestion)
                    if np.random.random() < 0.5:
                        current_ffs = (
                            self.config["params"]["links"]
                            .get(link_id, {})
                            .get("free_flow_speed", defaults["free_flow_speed"])
                        )
                        params["free_flow_speed"] = current_ffs * np.random.uniform(
                            0.6, 0.9
                        )

                    if params:
                        link_overrides[link_id] = params
                        # Explicitly set reverse link to ensure symmetry regardless of iteration order in create_network
                        u, v = link_id.split("_")
                        reverse_id = f"{v}_{u}"
                        link_overrides[reverse_id] = params.copy()

        return link_overrides


if __name__ == "__main__":
    data_manager = NetworkEnvGenerator()
    network = data_manager.create_network("delft")
    # Run simulation
    for t in range(1, data_manager.config["params"]["simulation_steps"]):
        network.network_loading(t)
