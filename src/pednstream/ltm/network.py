"""Module for defining the Network class, which represents a transportation network with nodes, links, and demand functions."""

import logging
from .node import Node
from .link import Link, Separator
from .od_manager import ODManager, DemandGenerator
from .path_finder import PathFinder
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass, field
from numpy import ndarray
from .node import Node, NodeConfig
from .link import LinkConfig, Link
from typing import List, Callable, Dict
from .od_manager import ODManager

@dataclass
class SimulationParameters:
    """Simulation parameters."""
    params: dict = field(default_factory=dict) # Dictionary to hold various simulation parameters, can be extended as needed.


@dataclass
class NetworkConfig:
    """Data class for collecting data related to the network configuration."""
    adjacency_matrix: ndarray # represent the interconnections between nodes in the network.
    links: List[LinkConfig] # List of configuration objects representing the links in the network
    nodes: List[NodeConfig] # List of configuration objects representing the nodes in the network
    origin_nodes: list # 
    destination_nodes: list = field(default_factory=list) # List of destination nodes, can be empty if not specified
    od_flows: dict = field(default_factory=dict) # Dictionary to hold origin-destination flow information, from yaml file
    positions: dict = field(default_factory=dict) # Dictionary to hold node positions, from yaml file
    demand : dict = field(default_factory=dict) # Dictionary to hold demand information, from yaml file
    # TODO: consider moving simulation steps to here, as a global parameter for the network, instead of being part of the link configuration. This would make it easier to manage and change simulation steps for the entire network.
@dataclass
class SimulationConfig:
    """Configuration for simulation runtime and parameters."""
    simulation_steps: int
    unit_time: int
    assign_flows_type: str
    seed: Optional[int]
    path_finder: dict
    demand_params: dict  # Original YAML demand section for generating runtime demand
    od_flows: dict       # OD flows mapping (origin, destination) -> flow

@dataclass
class NetworkConfig:
    """Configuration for the physical network topology."""
    adjacency_matrix: np.ndarray
    links: list          # List of LinkConfig
    nodes: list          # List of NodeConfig
    origin_nodes: list
    destination_nodes: list = field(default_factory=list)
    positions: dict = field(default_factory=dict)


class Network:
    """Class representing a transportation network."""

    def __init__(self, config: NetworkConfig):   
        """Initialize the Network with a configuration dictionary."""
        self._log_level: int = logging.INFO 
        self._verbose: bool = True 
        self.config = config
        # Stores nodes and links for the network
        self.nodes: Dict[int, Node] = {} # Dictionary to hold Node objects, keyed by node ID. Ids match the ones in the adjacency matrix.
        self.links: Dict[str, LinkConfig] = {} # Dictionary to hold LinkConfig objects, keyed by link ID

        self._controllers: List[int] = [] # List to hold IDs of nodes that are controllers, can be populated based on node configurations. Used by the PathFinder. TODO: consider implementing dependency inversion.
    
        self._controller_gaters: List[int] = [] # Info passed by the user. TODO: Consider removing it.  

    @property
    def controllers(self):
        """Nodes defines as controllers in the network."""
        if not self._controllers:
            result = []
            for node in self.nodes.values():
                if node.is_controller is True:
                    result.append(node.node_id)
            self._controllers = result
            self._controller_gaters = self._controllers.copy()
            return self._controllers
        else:
           return self._controllers
    
    @property
    def controller_gaters(self):
        """Get the list of controller gaters."""
        # TODO: consider removing it. 
        return self._controller_gaters
 
    @property
    def log_level(self):
        """Get the log level."""
        return self._log_level
    
    @log_level.setter
    def log_level(self, value: int) -> None:
        """Set the log level."""
        self._log_level = value

    @property
    def verbose(self):
        """Get the verbose flag."""
        return self._verbose    

    @verbose.setter
    def verbose(self, value: bool) -> None:
        """Set the verbose flag."""
        self._verbose = value

    @staticmethod
    def setup_logger(log_level=logging.INFO, log_dir=None):
        """Set up and configure logger."""
        if log_dir is None:
            log_dir = Path.cwd() / "outputs" / "logs"
        else:
            log_dir = Path(log_dir)

        # Create logs directory if it doesn't exist
        log_dir.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(__name__)

        # Only add handlers if the logger doesn't have any
        if not logger.handlers:
            # Configure logging format
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # File handler
            file_handler = logging.FileHandler(log_dir / "network.log")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            # Set level
            logger.setLevel(log_level)

        return logger
    
    # TODO: CONTINUE HERE
    def _create_nodes(self) -> dict[int, Node]:
        """Create Node objects based on the provided node configurations."""
        import numpy as np
        nodes = {}

        
        for node_config in self.config.nodes:
            
            # FIXME: could this be done by the node instantiation?
            id = node_config.id
            incoming_links = np.sum(self.config.adjacency_matrix[:, id])
            outgoing_links = np.sum(self.config.adjacency_matrix[id, :])

            # create node 
            node = Node(node_config)
       
            # Rules to set  note types
            if incoming_links >= 2 and outgoing_links >= 2:
                
                if id in self.config.origin_nodes or id in self.config.destination_nodes:
                    node.type = "regular"
                    # follow up steps:
                    #  1. Create virtual Link
                    # TODO: CONTINUE HERE: decide how to pass parametes to 
                    # create_virtual_links. 
                    node.create_virtual_links()

                    # 2. Assigne demand to the virtual link.
                    self._create_origin_destination(node_config) # replace this by the steps above
                else:  # do not create virtual links 
                    node.type = "onetoone"  
            elif incoming_links == 1 and outgoing_links ==1:
                node.type = "onetoone"
                # create virtual link
                self._create_origin_destination(node_config)
     
            # create the Node object
            node = Node(node_config)
            nodes[node.id] = node
        return nodes
    
    def _create_links(self) -> dict[tuple[int,int], Link]:
        """Create LinkConfig objects based on the provided link configurations."""
        links = {}
        for link_config in self.config.links:
            links[link_config.link_id] = link_config
        return links
    
    
    def create(self) -> None: 
        """Create the network based on the provided configuration and parameters."""
        nodes = self._create_nodes()
        links = self._create_links()

        # assemble the network




if __name__ == "__main__":
    # Example usage
    config = NetworkConfig(
        adjacency_matrix=None,  # Replace with actual adjacency matrix
        links=[],  # Replace with actual link configurations
        nodes=[],  # Replace with actual node configurations
        origin_nodes=[],  # Replace with actual origin nodes
        destination_nodes=[],  # Replace with actual destination nodes
        od_flows={},  # Replace with actual OD flows
        positions={}  # Replace with actual node positions
    )
    
    network = Network(config)
    network.create()