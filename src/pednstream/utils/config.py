"""Configuration interface.

It enables users to load condiguration parameters from a YAML file and validate them against the expected structure.
"""

import csv
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Mapping
from pednstream.exceptions import RequiredConfigError, InvalidConfigError

from dataclasses import dataclass, field

# Expected filenames simulations parameters when
# expected as separate file.
ADJ_MATRIX_FILENAME = "adj_matrix.npy" 
OD_FLOWS_FILENAME = "od_flows.csv"

class PathFinderParams:
    """Dataclass for path finder parameters."""

    def __init__(self, **kwargs):
        """Initialize PathFinderSpec with default values and override with kwargs."""
        for k, v in kwargs.items():
            setattr(self, k, v)


class DemandParams:
    """Dataclass for demand parameters."""

    def __init__(self, **kwargs):
        """Initialize DemandSpec with default values and override with kwargs."""
        for k, v in kwargs.items():
            setattr(self, k, v)

@dataclass(frozen=True)
class ScenarioParams:
    """Normilized scenario specification from YAML config. Contains all the parameters needed to initialize a Network instance and run a simulation."""

    origin_nodes: tuple[int, ...]
    destination_nodes: tuple[int, ...]
    simulation_steps: int
    unit_time: float
    default_link: Mapping[str, Any]
    link_overrides: Mapping[tuple[int, int], Mapping[str, Any]]
    demand: Mapping[int, DemandParams]
    controller_nodes: Optional[frozenset[int]]
    controller_links: Optional[frozenset[tuple[int, int]]]
    controller_schedule: Optional[Mapping[int, Any]] = field(default_factory=dict)
    seed: int | None = None
    assign_flows_type: str = "classic"
    path_finder: PathFinderParams = PathFinderParams()
    adjacency_matrix: np.ndarray | None = None
    od_flows: Optional[Mapping[tuple[int, int], float]] = None
    links: Optional[Mapping[tuple[int, int], Mapping[str, Any]]] = None

    def od_flows_from_file(self, path:Path) -> None:
        """Load OD flows from a long-format CSV with columns: origin, destination, weight.
        
        Args:
            path (Path): Path to the CSV file containing OD flows.
        """
        object.__setattr__(
            self,
            "od_flows",
            _load_od_flows_from_csv(
            path,
            origin_nodes=self.origin_nodes,
            destination_nodes=self.destination_nodes,
        ))
        return None

    def adjancency_matrix_from_file(self, path: Path) -> None:
        """Load adjacency matrix from a .npy file.
        
        Args:
            path (Path): Path to the .npy file containing the adjacency matrix.
        """
        object.__setattr__(
            self,
            "adjacency_matrix",
            np.load(Path(path).as_posix())
        )
        return None
    

def _load_od_flows_from_csv(
    csv_path: Path,
    origin_nodes: tuple[int, ...],
    destination_nodes: tuple[int, ...]
) -> Dict[tuple, float]:
    """Load OD flows from a long-format CSV with columns: origin, destination, weight.

    Pairs whose origin/destination are not in the declared node lists are
    silently filtered out. Zero weights are preserved.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"od_flows_csv file not found: {csv_path}")

    origin_set = set(origin_nodes)
    destination_set = set(destination_nodes)
    od_flows: Dict[tuple, float] = {}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        required_cols = {"origin", "destination", "weight"}
        if reader.fieldnames is None or not required_cols.issubset(reader.fieldnames):
            raise InvalidConfigError(
                f"od_flows_csv '{csv_path}' must have columns: "
                f"{sorted(required_cols)}; got {reader.fieldnames}"
            )

        for row in reader:
            o = int(row["origin"])
            d = int(row["destination"])
            if o not in origin_set or d not in destination_set:
                continue
            od_flows[(o, d)] = float(row["weight"])

    return od_flows



def fetch_data_files(path: str) -> Dict[str, Path]:
    """Fetches data files from a directory.
    
    Args:
        path: Path to the simulation data directory containing files like adj_matrix.npy, node_positions.json, od_flows.csv, etc.

    Returns:
        dict: A dictionary containing paths to files at the root of path.
    """
    data_dir = Path(path)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Simulation data directory not found: {data_dir}")
    
    # filter only files
    data_files = {f.name: f for f in data_dir.iterdir() if f.is_file()}
    
    return data_files


def read_scenario(path) -> ScenarioParams:
    """Read and validate a scenario configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        ScenarioSpec: A dataclass containing the normalized scenario parameters.
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Validate the configuration
    validate_config(config)

    # retrieve adjacency matrix from either YAML or .npy file
    data_files=fetch_data_files(str(Path(path).parent))
    if config["network"].get("adjacency_matrix") and ADJ_MATRIX_FILENAME in data_files:
        raise InvalidConfigError(
            F"Both 'adjacency_matrix' in YAML and {ADJ_MATRIX_FILENAME} file found; pick one."
        )
    elif config["network"].get("adjacency_matrix"): 
        adjacency_matrix=np.array(config["network"]["adjacency_matrix"])
    elif ADJ_MATRIX_FILENAME in data_files:
            adjacency_matrix=np.load(data_files[ADJ_MATRIX_FILENAME].as_posix())
    else:
            adjacency_matrix=None

    # Initialize ScenarioSpec dataclass
    scenario_spec = ScenarioParams(
        origin_nodes=tuple(config["network"]["origin_nodes"]),
        destination_nodes=tuple(config["network"].get("destination_nodes", [])),
        simulation_steps=config["simulation"]["simulation_steps"],
        unit_time=config["simulation"]["unit_time"],
        default_link=config["default_link"],
        link_overrides={
            _parse_pair_key(k, label="links", example="i_j"): v
            for k, v in config.get("links", {}).items()
        },
        demand={
            int(k.split("_")[1]): DemandParams(**v)
            for k, v in config.get("demand", {}).items()
        },
        od_flows={
            _parse_pair_key(k, label="od_flows", example="origin_destination"): v
            for k, v in config.get("od_flows", {}).items()
        },
        controller_nodes=frozenset(
            map(int, config.get("controllers", {}).get("nodes", []))
        ),
        controller_links=frozenset(
            _parse_pair_key(k, label="controllers.links", example="i_j")
            for k in config.get("controllers", {}).get("links", [])
        ),
        seed=config["simulation"].get("seed"),
        assign_flows_type=config["simulation"].get("assign_flows_type", "classic"),
        path_finder=PathFinderParams(**config["simulation"].get("path_finder", {})),
    )

    # Load adjacency matrix if available
    if adjacency_matrix is not None:
        scenario_spec.adjancency_matrix_from_file(data_files[ADJ_MATRIX_FILENAME])

    # Load OD flows from CSV if specified
    if "od_flows_csv" in config:
        scenario_spec.od_flows_from_file(data_files[OD_FLOWS_FILENAME])
    
    return scenario_spec


# TODO: continue, move to validations and refactor if needed.
# CHECKS:

def _validate_od_flows(config: Dict[str, Any]) -> None:
    """Validate OD-flow key format and node membership."""
    if "od_flows" not in config:
        return

    origin_nodes = config["network"].get("origin_nodes", [])
    destination_nodes = config["network"].get("destination_nodes", [])

    if not destination_nodes:
        raise InvalidConfigError(
            "od_flows requires destination_nodes to be defined and non-empty"
        )

    for od_pair in config["od_flows"].keys():
        origin, dest = _parse_pair_key(
            od_pair,
            label="od_flows",
            example="origin_destination (e.g., '1_2')",
        )

        if origin not in origin_nodes:
            raise InvalidConfigError(
                f"od_flows key '{od_pair}': origin {origin} is not in origin_nodes {origin_nodes}"
            )

        if dest not in destination_nodes:
            raise InvalidConfigError(
                f"od_flows key '{od_pair}': destination {dest} is not in destination_nodes {destination_nodes}"
            )


def _validate_od_flows_csv(config: Dict[str, Any]) -> None:
    """Validate that od_flows_csv is not used together with od_flows.

    Also validates that destination_nodes is defined.
    """
    if "od_flows_csv" not in config:
        return

    if "od_flows" in config:
        raise InvalidConfigError(
            "Cannot specify both 'od_flows' and 'od_flows_csv'; pick one"
        )

    destination_nodes = config["network"].get("destination_nodes", [])
    if not destination_nodes:
        raise InvalidConfigError(
            "od_flows_csv requires destination_nodes to be defined and non-empty"
        )





def _validate_demand(config: Dict[str, Any]) -> None:
    """Validate demand key format and origin membership."""
    if "demand" not in config:
        return

    origin_nodes = config["network"].get("origin_nodes", [])

    for demand_key in config["demand"].keys():
        if not demand_key.startswith("origin_"):
            raise InvalidConfigError(
                f"Invalid demand key format: '{demand_key}'. Expected 'origin_X' (e.g., 'origin_0')"
            )

        try:
            origin_id = int(demand_key.split("_")[1])
        except (IndexError, ValueError) as exc:
            raise InvalidConfigError(
                f"Invalid demand key format: '{demand_key}'. Expected 'origin_X' where X is an integer"
            ) from exc

        if origin_id not in origin_nodes:
            raise InvalidConfigError(
                f"demand key '{demand_key}': origin {origin_id} is not in origin_nodes {origin_nodes}"
            )


def _validate_links(
    config: Dict[str, Any], adjacency_matrix: Optional[np.ndarray] = None
) -> None:
    """Validate link key format and edge existence when adjacency is available."""
    if "links" not in config or adjacency_matrix is None:
        return

    for link_key in config["links"].keys():
        i, j = _parse_pair_key(link_key, label="links", example="i_j (e.g., '1_2')")

        if i >= adjacency_matrix.shape[0] or j >= adjacency_matrix.shape[1]:
            raise InvalidConfigError(
                f"links key '{link_key}': node index out of bounds for adjacency matrix of shape {adjacency_matrix.shape}"
            )

        if adjacency_matrix[i, j] != 1:
            raise InvalidConfigError(
                f"links key '{link_key}': no edge exists between nodes {i} and {j} in adjacency matrix"
            )

def validate_config(
    config: Dict[str, Any], adjacency_matrix: Optional[np.ndarray] = None
) -> None:
    """Validate configuration parameters.

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
    _validate_od_flows(config)
    _validate_od_flows_csv(config)
    _validate_demand(config)
    _validate_links(config, adjacency_matrix=adjacency_matrix)
    return None


# BUILDS:


def build_configs(spec, adjacency_matrix, positions=None) -> tuple[NetworkConfig, SimulationConfig]:
    """Build NetworkConfig and SimulationConfig dataclasses from the ScenarioSpec and adjacency matrix.

    Args:
        spec: ScenarioSpec dataclass containing the normalized scenario parameters.
        adjacency_matrix: NxN numpy array representing the network topology.

    """
     # TODO: Implement

    return None, None



def _assemble_network_config(params: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble the network configuration dictionary."""
    network_config = {
        "params": params,
        "origin_nodes": config["network"]["origin_nodes"],
        "destination_nodes": config["network"].get("destination_nodes", []),
    }

    if "adjacency_matrix" in config["network"]:
        network_config["adjacency_matrix"] = np.array(
            config["network"]["adjacency_matrix"]
        )
    else: # assign None
        network_config["adjacency_matrix"] = None

    return network_config

def _build_link_configs(
    adjacency_matrix: np.ndarray,
    default_link: Dict[str, Any],
    links_overrides: Dict[str, Any],
    simulation_steps: int,
    unit_time: int,
    controller_links: list = [],
) -> list:
    """Build a list of LinkConfig objects from the adjacency matrix.

    For each edge (i, j) in the adjacency matrix, creates both a forward (i→j)
    and reverse (j→i) LinkConfig. Uses link-specific overrides when available,
    otherwise falls back to default_link parameters.

    Args:
        adjacency_matrix: NxN numpy array of 0s and 1s.
        default_link: Default link parameters from YAML (length, width, etc.)
        links_overrides: Link-specific overrides, keyed by "i_j" strings.
        simulation_steps: Number of simulation steps.
        unit_time: Unit time for the simulation.
        controller_links: List of controller link keys (e.g. ["1-2"]).

    Returns:
        List of LinkConfig dataclass instances.
    """
    from pednstream.ltm.link import LinkConfig

    link_configs = []
    num_nodes = adjacency_matrix.shape[0]

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency_matrix[i, j] != 1:
                continue

            # Resolve parameters: check both "i_j" and "j_i" keys
            forward_key = f"{i}_{j}"
            reverse_key = f"{j}_{i}"
            if forward_key in links_overrides:
                merged_params = {**default_link, **links_overrides[forward_key]}
            elif reverse_key in links_overrides:
                merged_params = {**default_link, **links_overrides[reverse_key]}
            else:
                merged_params = dict(default_link)

            # Determine if this is a controller (separator) link
            is_controller = (
                f"{i}-{j}" in controller_links
                or f"{j}-{i}" in controller_links
            )

            # Build forward LinkConfig (i → j)
            forward_config = LinkConfig(
                start_node=i,
                end_node=j,
                link_id=(i, j),
                simulation_steps=simulation_steps,
                unit_time=unit_time,
                is_controller=is_controller,
                length=merged_params["length"],
                width=merged_params["width"],
                free_flow_speed=merged_params["free_flow_speed"],
                k_critical=merged_params["k_critical"],
                k_jam=merged_params["k_jam"],
                gamma=merged_params.get("gamma", 2e-3),
                bi_factor=merged_params.get("bi_factor", 1),
                fd_type=merged_params.get("fd_type", "yperman"),
                speed_noise_std=merged_params.get("speed_noise_std", 0),
                activity_probability=merged_params.get("activity_probability", 0.0),
                front_gate_width=merged_params.get("front_gate_width"),
                back_gate_width=merged_params.get("back_gate_width"),
            )

            # Build reverse LinkConfig (j → i) — swap front/back gate widths
            reverse_config = LinkConfig(
                start_node=j,
                end_node=i,
                link_id=(j, i),
                simulation_steps=simulation_steps,
                unit_time=unit_time,
                is_controller=is_controller,
                length=merged_params["length"],
                width=merged_params["width"],
                free_flow_speed=merged_params["free_flow_speed"],
                k_critical=merged_params["k_critical"],
                k_jam=merged_params["k_jam"],
                gamma=merged_params.get("gamma", 2e-3),
                bi_factor=merged_params.get("bi_factor", 1),
                fd_type=merged_params.get("fd_type", "yperman"),
                speed_noise_std=merged_params.get("speed_noise_std", 0),
                activity_probability=merged_params.get("activity_probability", 0.0),
                front_gate_width=merged_params.get("back_gate_width"),   # swapped
                back_gate_width=merged_params.get("front_gate_width"),   # swapped
            )

            link_configs.append(forward_config)
            link_configs.append(reverse_config)

    return link_configs


def _build_node_configs(
    adjacency_matrix: np.ndarray,
    origin_nodes: list,
    destination_nodes: list,
    demand_config: Dict[str, Any] = {},
    controller_nodes: set = set(),
) -> list:
    """Build a list of NodeConfig objects from the adjacency matrix.

    Determines node_type ("onetoone" or "regular") based on connection counts
    and origin/destination membership, mirroring Network._create_nodes() logic.
    For origin nodes, parses the demand profile from the YAML config.

    Args:
        adjacency_matrix: NxN numpy array of 0s and 1s.
        origin_nodes: List of origin node IDs.
        destination_nodes: List of destination node IDs.
        demand_config: The "demand" section from the YAML config, e.g.
            {"origin_0": {"pattern": "sudden_demand", "peak_lambda": 20, ...}}.
        controller_nodes: Optional set of node IDs that act as controllers.

    Returns:
        List of NodeConfig dataclass instances.
    """
    from pednstream.ltm.node import NodeConfig

    if controller_nodes is None:
        controller_nodes = set()
    if demand_config is None:
        demand_config = {}

    node_configs = []
    num_nodes = adjacency_matrix.shape[0]

    for node_id in range(num_nodes):
        incoming_count = int(np.sum(adjacency_matrix[:, node_id]))
        outgoing_count = int(np.sum(adjacency_matrix[node_id, :]))

        is_od = node_id in origin_nodes or node_id in destination_nodes

        # Determine node_type — same logic as Network._create_nodes()
        if incoming_count == 2 and outgoing_count == 2:
            node_type = "regular" if is_od else "onetoone"
        elif incoming_count == 1 and outgoing_count == 1:
            node_type = "onetoone"
        else:
            node_type = "regular"
            
        node_config = NodeConfig(
            id=node_id,
            type=node_type
        )
        node_configs.append(node_config)

    return node_configs


def _new_assemble_network_config(config: Dict[str, Any], adjacency_matrix: np.ndarray | None = None) -> tuple["NetworkConfig", "SimulationConfig"]:
    """Assemble NetworkConfig and SimulationConfig dataclasses from the raw YAML config dictionary.

    Args:
        config: The raw parsed YAML configuration dictionary.
        adjacency_matrix: Optional adjacency matrix. If None, attempts to read from config.

    Returns:
        tuple: (NetworkConfig, SimulationConfig) dataclass instances.
    """
    from pednstream.ltm.network import NetworkConfig, SimulationConfig

    # --- Simulation parameters ---
    path_finder_params = config["simulation"].get("path_finder", {})
    simulation_steps = config["simulation"]["simulation_steps"]
    unit_time = config["simulation"]["unit_time"]

    # --- Adjacency matrix ---
    if adjacency_matrix is None and "adjacency_matrix" in config.get("network", {}):
        adjacency_matrix = np.array(config["network"]["adjacency_matrix"])

    # --- Build LinkConfig list ---
    link_configs = []
    if adjacency_matrix is not None:
        controller_links = config.get("controllers", {}).get("links", [])
        link_configs = _build_link_configs(
            adjacency_matrix=adjacency_matrix,
            default_link=config["default_link"],
            links_overrides=config.get("links", {}),
            simulation_steps=simulation_steps,
            unit_time=unit_time,
            controller_links=controller_links,
        )

    # --- Build NodeConfig list ---
    origin_nodes = config["network"]["origin_nodes"]
    destination_nodes = config["network"].get("destination_nodes", [])
    node_configs = []
    if adjacency_matrix is not None:
        controller_nodes = set(
            map(int, config.get("controllers", {}).get("nodes", []))
        )
        node_configs = _build_node_configs(
            adjacency_matrix=adjacency_matrix,
            origin_nodes=origin_nodes,
            destination_nodes=destination_nodes,
            demand_config=config.get("demand", {}),
            controller_nodes=controller_nodes,
        )

    # --- OD flows (inline dict) ---
    od_flows = {}
    if "od_flows" in config:
        for od_pair, flow in config["od_flows"].items():
            origin, dest = map(int, od_pair.split("_"))
            od_flows[(origin, dest)] = flow

    # --- Build Config Objects ---
    network_config = NetworkConfig(
        adjacency_matrix=adjacency_matrix,
        links=link_configs,
        nodes=node_configs,
        origin_nodes=origin_nodes,
        destination_nodes=destination_nodes,
        positions={},   # populated later from node_positions.json
    )

    simulation_config = SimulationConfig(
        simulation_steps=simulation_steps,
        unit_time=unit_time,
        assign_flows_type=config["simulation"].get("assign_flows_type", "classic"),
        seed=config["simulation"].get("seed", None),
        path_finder=path_finder_params,
        demand_params=config.get("demand", {}),
        od_flows=od_flows,
    )

    return network_config, simulation_config


def _parse_pair_key(key: str, label: str, example: str) -> tuple[int, int]:
    """Parse keys in the form 'i_j' and return integer tuple (i, j)."""
    try:
        parts = key.split("_")
        if len(parts) != 2:
            raise ValueError
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise InvalidConfigError(
            f"Invalid {label} key format: '{key}'. Expected '{example}'"
        ) from exc



def load_config(config_path: str) -> dict:
    """Load and validate configuration from a YAML file with a flattened structure.

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
    network_config = _assemble_network_config(params, config)

    # 4. Handle optional 'od_flows' (inline YAML dict) or 'od_flows_csv' (CSV file)
    if "od_flows" in config and "od_flows_csv" in config:
        raise InvalidConfigError(
            "Cannot specify both 'od_flows' and 'od_flows_csv'; pick one"
        )

    if "od_flows" in config:
        od_flows = {}
        for od_pair, flow in config["od_flows"].items():
            origin, dest = map(int, od_pair.split("_"))
            od_flows[(origin, dest)] = flow
        network_config["od_flows"] = od_flows
    elif "od_flows_csv" in config:
        csv_path = Path(config_path).parent / config["od_flows_csv"]
        network_config["od_flows"] = _load_od_flows_from_csv(
            csv_path,
            origin_nodes=config["network"].get("origin_nodes", []),
            destination_nodes=config["network"].get("destination_nodes", []),
        )

    return network_config




if __name__ == "__main__":
    # Example usage
    config_path = "tests/data/delft/sim_params.yaml"
    # config = load_config(config_path)
    config = read_scenario(config_path)
    # validate_config(config)
    print(config)

    files = fetch_data_files("tests/data/delft")
    # print(files)