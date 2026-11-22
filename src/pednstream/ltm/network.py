"""Module for defining the Network class, which represents a transportation network with nodes, links, and demand functions."""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pathlib import Path
from numpy import ndarray
from pednstream.ltm.node import Node, NodeConfig
from pednstream.ltm.link import LinkConfig, Link, RegularCreator, SeparatorCreator

@dataclass
class NetworkConfig:
    """Configuration for the physical network topology."""
    adjacency_matrix: ndarray
    links: list[LinkConfig]          # List of LinkConfig
    nodes: list[NodeConfig]          # List of NodeConfig
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
        self.links: Dict[tuple[int, int], Link] = {} # Dictionary to hold Link objects, keyed by link ID (start, end)
        self._simulation_steps: Optional[int] = None # Set by allocate(); the topology does not depend on it.
        self._controllers: List[int] = [] # List to hold IDs of nodes that are controllers, can be populated based on node configurations. Used by the PathFinder. TODO: consider implementing dependency inversion.
        self._controller_gaters: List[int] = [] # Info passed by the user. TODO: Consider removing it.  

    @property
    def controllers(self):
        """Nodes defines as controllers in the network."""
        if not self._controllers:
            result = []
            for node in self.nodes.values():
                if node.is_controller is True:
                    result.append(node.id)
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
    def simulation_steps(self) -> int:
        """Number of steps the network state is currently allocated for."""
        if self._simulation_steps is None:
            raise RuntimeError(
                "Network state is not allocated; call Network.allocate(simulation_steps) first."
            )
        return self._simulation_steps

    @property
    def is_allocated(self) -> bool:
        """Whether the time-series state has been allocated."""
        return self._simulation_steps is not None

    @property
    def log_level(self) -> int:
        """Get the log level."""
        return self._log_level
    
    @log_level.setter
    def log_level(self, value: int) -> None:
        """Set the log level."""
        self._log_level = value

    @property
    def verbose(self) -> bool:
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
    
    def _create_nodes(self) -> dict[int, Node]:
        """Create Node objects (topology only) based on the provided node configurations.

        Origin/destination nodes get a pair of virtual links. No simulation state is
        allocated here, see allocate().
        """
        import numpy as np
        nodes = {}
        od_nodes = set(self.config.origin_nodes) | set(self.config.destination_nodes)

        for node_config in self.config.nodes:
            node_id = node_config.id
            incoming_links = np.sum(self.config.adjacency_matrix[:, node_id])
            outgoing_links = np.sum(self.config.adjacency_matrix[node_id, :])
            is_od = node_id in od_nodes

            node = Node(node_config)

            # Node type rules (same as depricated_network.Network._create_nodes).
            # FIXME: the type setter on Node expects a tuple; types should come from NodeConfig (see node.py).
            if incoming_links == 2 and outgoing_links == 2:
                node._type = "regular" if is_od else "onetoone"
                create_virtual = is_od
            elif incoming_links == 1 and outgoing_links == 1:
                node._type = "onetoone"
                create_virtual = True
            else:
                node._type = "regular"
                create_virtual = is_od

            if create_virtual:
                node.create_virtual_links()

            nodes[node.id] = node
        return nodes

    def _create_links(self) -> dict[tuple[int, int], Link]:
        """Create Link objects based on the provided link configurations."""
        links = {}
        for link_config in self.config.links:
            creator = SeparatorCreator() if link_config.is_controller else RegularCreator()
            links[link_config.link_id] = creator.create_link(link_config)

        # Pair forward and reverse links
        for (i, j), link in links.items():
            reverse = links.get((j, i))
            if reverse is not None:
                link.reverse_link = reverse
        return links

    def create(self) -> None:
        """Create the network topology (nodes and links).

        The time horizon is not needed here. Call allocate(simulation_steps) before
        running a simulation.
        """
        self.nodes = self._create_nodes()
        self.links = self._create_links()

        # assemble the network: attach physical links to their nodes
        for (i, j), link in self.links.items():
            self.nodes[i].outgoing_links.append(link)
            self.nodes[j].incoming_links.append(link)

        for node in self.nodes.values():
            node.init_mask()
        return None

    def _all_links(self):
        """Yield physical and virtual links of the network."""
        yield from self.links.values()
        for node in self.nodes.values():
            for v_link in (node.virtual_incoming_link, node.virtual_outgoing_link):
                if v_link is not None:
                    yield v_link

    def allocate(self, simulation_steps: int) -> None:
        """Allocate (or reset) the time-series state of all links and nodes for a run.

        Calling it again discards previous results, so one network can be reused for
        several simulations, possibly with different horizons. Demand is reset to zeros;
        assign origin demand afterwards with set_demand().

        Args:
            simulation_steps (int): number of simulation steps of the run.
        """
        import numpy as np
        if not self.nodes:
            raise RuntimeError("Network is empty; call Network.create() before allocate().")

        for link in self._all_links():
            link.allocate_state(simulation_steps)

        for node in self.nodes.values():
            if node.virtual_incoming_link is not None:
                node.demand = np.zeros(simulation_steps + 1)

        self._simulation_steps = simulation_steps
        return None

    def set_demand(self, node_id: int, demand: ndarray) -> None:
        """Assign the demand profile of an origin node for the allocated run.

        Args:
            node_id (int): id of an origin node (it must have virtual links).
            demand (ndarray): demand per time step, of length simulation_steps or
                simulation_steps + 1 (DemandGenerator patterns produce either). It is
                stored zero-padded to simulation_steps + 1, like the link arrays.
        """
        import numpy as np
        steps = self.simulation_steps  # raises if not allocated
        node = self.nodes[node_id]
        if node.virtual_incoming_link is None:
            raise ValueError(f"Node {node_id} has no virtual links; it is not an origin node.")
        demand = np.asarray(demand, dtype=float)
        if demand.ndim != 1 or len(demand) not in (steps, steps + 1):
            raise ValueError(
                f"Demand for node {node_id} must be 1-D with length {steps} or {steps + 1} "
                f"(simulation_steps = {steps}), got shape {demand.shape}."
            )
        node.demand = np.pad(demand, (0, steps + 1 - len(demand)))
        return None


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