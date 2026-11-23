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
        self._loading_order: List[int] = [] # Order in which nodes are processed each step, see _discovery_order().
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

        adjacency_matrix = self.config.adjacency_matrix
        for node_config in self.config.nodes:
            node_id = node_config.id
            incoming_links = np.sum(adjacency_matrix[:, node_id])
            outgoing_links = np.sum(adjacency_matrix[node_id, :])
            is_od = node_id in od_nodes

            # A dead end is a node with a single neighbour. Count distinct neighbours,
            # not links: "1 in, 1 out" is a dead end only when both links go to the
            # same neighbour (0 ⇄ 1), not in a pass-through like 0 → 1 → 2.
            # Dead ends need virtual links: U-turns are not allowed by the node model,
            # so without them pedestrians reaching a dead end could never leave it
            # (the node would have zero turning movements).
            neighbours = set(np.flatnonzero(adjacency_matrix[node_id, :])) | set(
                np.flatnonzero(adjacency_matrix[:, node_id])
            )
            is_dead_end = len(neighbours) == 1

            node = Node(node_config)

            # Node type rules (same as depricated_network.Network._create_nodes).
            # FIXME: the type setter on Node expects a tuple; types should come from NodeConfig (see node.py).
            if incoming_links == 2 and outgoing_links == 2:
                node._type = "regular" if is_od else "onetoone"
            elif incoming_links == 1 and outgoing_links == 1:
                node._type = "onetoone"
            else:
                node._type = "regular"

            # Virtual links act as entry/exit points: for origins/destinations and dead ends.
            if is_od or is_dead_end:
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

        Raises:
            ValueError: if the adjacency matrix is not symmetric. Every link needs a
                reverse link, since the node model reads the reverse link's sending flow.
        """
        import numpy as np
        adjacency_matrix = np.asarray(self.config.adjacency_matrix)
        if not np.array_equal(adjacency_matrix, adjacency_matrix.T):
            raise ValueError(
                "Adjacency matrix must be symmetric: every link needs a reverse link."
            )

        self.nodes = self._create_nodes()
        self.links = self._create_links()

        # assemble the network: attach physical links to their nodes
        for (i, j), link in self.links.items():
            self.nodes[i].outgoing_links.append(link)
            self.nodes[j].incoming_links.append(link)

        for node in self.nodes.values():
            node.init_mask()

        self._loading_order = self._discovery_order(adjacency_matrix)
        return None

    @staticmethod
    def _discovery_order(adjacency_matrix) -> List[int]:
        """Order in which network_loading processes the nodes.

        Nodes are visited in the order the reference implementation (92fa2a3 /
        v0.0.2-alpha) created them while scanning the upper triangle of the adjacency
        matrix: node i, then each not yet seen neighbour j > i. Each node draws random
        numbers when assigning flows, so this order keeps seeded runs reproducible.
        """
        order: List[int] = []
        seen = set()
        num_nodes = adjacency_matrix.shape[0]
        for i in range(num_nodes):
            if i not in seen:
                seen.add(i)
                order.append(i)
            for j in range(i + 1, num_nodes):
                if adjacency_matrix[i, j] == 1 and j not in seen:
                    seen.add(j)
                    order.append(j)
        return order

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

    def update_link_states(self, time_step: int) -> None:
        """Update densities, then speeds, of all physical links for a time step.

        All densities are updated before any speed, because a link's speed depends on
        the density of its reverse link.
        """
        for link in self.links.values():
            link.update_density_flow(time_step)
        for link in self.links.values():
            link.update_speeds(time_step)
        return None

    def network_loading(
        self,
        time_step: int,
        path_finder=None,
        od_manager=None,
        assign_flows_type: str = "classic",
    ) -> None:
        """Advance the network by one time step (node model, then link states).

        Args:
            time_step (int): current time step, from 1 to simulation_steps.
            path_finder (PathFinder | None): route choice; when given with an
                od_manager, turning fractions are updated from the OD flows and only
                nodes on OD paths assign flows. Without it every node assigns flows
                with its current turning fractions.
            od_manager (ODManager | None): OD flows used by the path finder.
            assign_flows_type (str): solver for regular nodes, "classic" or "optimal".
        """
        import numpy as np
        steps = self.simulation_steps  # raises if not allocated
        if not 1 <= time_step <= steps:
            raise ValueError(f"time_step must be between 1 and {steps}, got {time_step}.")

        for node_id in self._loading_order:
            node = self.nodes[node_id]
            if node.turning_fractions is None:  # default: equal split
                phi = 1 / (node.dest_num - 1)
                node.turning_fractions = np.ones(node.edge_num) * phi

            if path_finder is not None and od_manager is not None:
                path_finder.calculate_node_turning_fractions(
                    time_step=time_step, od_manager=od_manager, node=node
                )

            if path_finder is None or node.id in path_finder.nodes_in_paths:
                if node.type == "onetoone":
                    node.assign_flows(time_step)
                else:  # regular node
                    if node.A_ub is None:
                        node.get_matrix_A()
                    node.assign_flows(time_step, type=assign_flows_type)

        self.update_link_states(time_step)
        return None


if __name__ == "__main__":
    import numpy as np

    # Example usage
    # A simple 3-node corridor: 0 ⇄ 1 ⇄ 2. Every street is two links, one per direction,
    # so the adjacency matrix is symmetric.
    adjacency_matrix = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])

    link_configs = [
        LinkConfig(
            link_id=(i, j),
            start_node=i,
            end_node=j,
            unit_time=1.0,
            length=100.0,
            width=3.5,
            free_flow_speed=15.0,
            k_critical=20.0,
            k_jam=140.0
        )
        for i, j in [(0, 1), (1, 0), (1, 2), (2, 1)]
    ]

    node_configs = [
        NodeConfig(id=0, type="origin", is_controller=False),
        NodeConfig(id=1, type="regular", is_controller=False),
        NodeConfig(id=2, type="destination", is_controller=False)
    ]

    positions = {0: (0, 0), 1: (100, 0), 2: (200, 0)}

    config = NetworkConfig(
        adjacency_matrix=adjacency_matrix,
        links=link_configs,
        nodes=node_configs,
        origin_nodes=[0],
        destination_nodes=[2],
        positions=positions
    )

    # create(), allocate() and set_demand() modify the network in place and return None.
    network = Network(config)
    network.create()
    network.allocate(simulation_steps=100)
    network.set_demand(node_id=0, demand=[10] * 101)

    with_virtual = [nid for nid, n in network.nodes.items() if n.virtual_incoming_link is not None]
    print(f"links: {list(network.links)}")
    print(f"nodes with virtual links: {with_virtual}")  # [0, 2]: the two dead ends
