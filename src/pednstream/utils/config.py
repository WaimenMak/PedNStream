import yaml
import numpy as np
from typing import Dict, Any, Optional
from pednstream.exceptions import RequiredConfigError, InvalidConfigError


def load_config(config_path: str) -> dict:
    """
    Load and validate configuration from a YAML file with a flattened structure.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        dict: Configuration parameters for the Network class
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 1. Assemble the 'params' dictionary required by the Network class
    path_finder_params = config["simulation"].get("path_finder", {})

    params = {
        "simulation_steps": config["simulation"]["simulation_steps"],
        "unit_time": config["simulation"]["unit_time"],
        "assign_flows_type": config["simulation"].get("assign_flows_type", "classic"),
        "seed": config["simulation"].get("seed", None),
        "path_finder": path_finder_params,
        "default_link": config["default_link"],
        "links": config.get("links", {}),
        "demand": config.get("demand", {}),
        "controllers": config.get("controllers", {}),
    }

    # 2. Assemble the config dictionary
    network_config = {
        "params": params,
        "origin_nodes": config["network"]["origin_nodes"],
        "destination_nodes": config["network"].get("destination_nodes", []),
    }

    if "adjacency_matrix" in config["network"]:
        network_config["adjacency_matrix"] = np.array(
            config["network"]["adjacency_matrix"]
        )

    # 4. Handle optional 'od_flows'
    if "od_flows" in config:
        od_flows = {}
        for od_pair, flow in config["od_flows"].items():
            origin, dest = map(int, od_pair.split("_"))
            od_flows[(origin, dest)] = flow
        network_config["od_flows"] = od_flows

    return network_config


def validate_config(
    config: Dict[str, Any], adjacency_matrix: Optional[np.ndarray] = None
) -> None:
    """
    Validate configuration parameters

    Args:
        config: Configuration dictionary to validate
        adjacency_matrix: Optional adjacency matrix for validating link overrides

    Raises:
        RequiredConfigError: If required section/field is missing
        InvalidConfigError: If configuration value is invalid
    """
    required_fields = {
        "network": ["origin_nodes"],
        "simulation": ["simulation_steps", "unit_time"],
        "default_link": ["length", "width", "free_flow_speed", "k_critical", "k_jam"],
    }

    for section, fields in required_fields.items():
        if section not in config:
            raise RequiredConfigError(
                f"Missing required section in configuration: {section}"
            )

        for field in fields:
            if field not in config[section]:
                raise RequiredConfigError(
                    f"Missing required field in configuration: {field} in section {section}"
                )

    # Validate od_flows if present
    if "od_flows" in config:
        origin_nodes = config["network"].get("origin_nodes", [])
        destination_nodes = config["network"].get("destination_nodes", [])

        if not destination_nodes:
            raise InvalidConfigError(
                "od_flows requires destination_nodes to be defined and non-empty"
            )

        for od_pair in config["od_flows"].keys():
            try:
                origin, dest = map(int, od_pair.split("_"))
            except ValueError:
                raise InvalidConfigError(
                    f"Invalid od_flows key format: '{od_pair}'. Expected 'origin_destination' (e.g., '1_2')"
                )

            if origin not in origin_nodes:
                raise InvalidConfigError(
                    f"od_flows key '{od_pair}': origin {origin} is not in origin_nodes {origin_nodes}"
                )

            if dest not in destination_nodes:
                raise InvalidConfigError(
                    f"od_flows key '{od_pair}': destination {dest} is not in destination_nodes {destination_nodes}"
                )

    # Validate demand if present
    if "demand" in config:
        origin_nodes = config["network"].get("origin_nodes", [])

        for demand_key in config["demand"].keys():
            if not demand_key.startswith("origin_"):
                raise InvalidConfigError(
                    f"Invalid demand key format: '{demand_key}'. Expected 'origin_X' (e.g., 'origin_0')"
                )

            try:
                origin_id = int(demand_key.split("_")[1])
            except (IndexError, ValueError):
                raise InvalidConfigError(
                    f"Invalid demand key format: '{demand_key}'. Expected 'origin_X' where X is an integer"
                )

            if origin_id not in origin_nodes:
                raise InvalidConfigError(
                    f"demand key '{demand_key}': origin {origin_id} is not in origin_nodes {origin_nodes}"
                )

    # Validate links if present and adjacency_matrix is provided
    if "links" in config and adjacency_matrix is not None:
        for link_key in config["links"].keys():
            try:
                i, j = map(int, link_key.split("_"))
            except ValueError:
                raise InvalidConfigError(
                    f"Invalid links key format: '{link_key}'. Expected 'i_j' (e.g., '1_2')"
                )

            if i >= adjacency_matrix.shape[0] or j >= adjacency_matrix.shape[1]:
                raise InvalidConfigError(
                    f"links key '{link_key}': node index out of bounds for adjacency matrix of shape {adjacency_matrix.shape}"
                )

            if adjacency_matrix[i, j] != 1:
                raise InvalidConfigError(
                    f"links key '{link_key}': no edge exists between nodes {i} and {j} in adjacency matrix"
                )
