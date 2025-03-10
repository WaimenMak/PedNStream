import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from .node import Node, OneToOneNode, RegularNode
from .link import Link
from .od_manager import ODManager
from .path_finder import PathFinder
from typing import Callable, Dict, Optional, List
import logging
import os

"""
A Link Transmission Model for the Pedestrian Traffic
"""

class DemandGenerator:
    def __init__(self, simulation_steps: int, params: dict, logger: logging.Logger):
        self.logger = logger
        self.simulation_steps = simulation_steps
        self.params = params
        self.time = np.arange(simulation_steps)
        
        # Dictionary to store built-in and custom demand patterns
        self.demand_patterns: Dict[str, Callable] = {
            'gaussian_peaks': self.generate_gaussian_peaks,
            'constant': self.generate_constant,
        }

    def register_pattern(self, pattern_name: str, pattern_func: Callable):
        """
        Register a new demand pattern function.
        
        Args:
            pattern_name: Name of the pattern to register
            pattern_func: Function that takes (origin_id, params) and returns np.array
        """
        if not callable(pattern_func):
            raise ValueError("pattern_func must be callable")
        self.demand_patterns[pattern_name] = pattern_func

    def generate_gaussian_peaks(self, origin_id: int, params=None) -> np.ndarray:
        """Built-in gaussian peaks demand pattern"""
        if params is None:
            params = self.params
            
        try:
            peak_lambda = params['demand'][f'origin_{origin_id}']['peak_lambda']
            base_lambda = params['demand'][f'origin_{origin_id}']['base_lambda']
        except KeyError:
            self.logger.info(f"No demand configuration found for origin {origin_id}, automatically set to default")
            peak_lambda = 10
            base_lambda = 5
            
        t = self.simulation_steps
        
        morning_peak = peak_lambda * np.exp(-(self.time - t/4)**2 / (2 * (t/20)**2))
        evening_peak = peak_lambda * np.exp(-(self.time - 3*t/4)**2 / (2 * (t/20)**2))
        lambda_t = base_lambda + morning_peak + evening_peak
        
        seed = self.params.get(f'seed', 42)
        np.random.seed(seed)
        
        demand = np.random.poisson(lam=lambda_t)
        
        return demand

    def generate_constant(self, origin_id: int, params=None) -> np.ndarray:
        """Built-in constant demand pattern"""
        # demand_value = self.params.get(f'demand.origin_{origin_id}.base_lambda', 10)
        demand_value = self.params['demand'][f'origin_{origin_id}']['base_lambda']
        return np.ones(self.simulation_steps) * demand_value

    def generate_custom(self, origin_id: int, pattern: str) -> np.ndarray:
        """
        Generate demand based on specified pattern name.
        
        Args:
            origin_id: ID of the origin node
            pattern: Name of the pattern to use
            
        Returns:
            np.ndarray: Generated demand array
        
        Raises:
            ValueError: If pattern name is not registered
        """
        if pattern not in self.demand_patterns:
            raise ValueError(f"Unknown demand pattern: {pattern}. "
                           f"Available patterns: {list(self.demand_patterns.keys())}")
        
        return self.demand_patterns[pattern](origin_id, params=self.params)


class Network:
    @staticmethod
    def setup_logger(log_level=logging.INFO, log_dir=os.path.join("..", "outputs", "logs")):
        """Set up and configure logger"""
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        logger = logging.getLogger(__name__)
        
        # Only add handlers if the logger doesn't have any
        if not logger.handlers:
            # Configure logging format
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # File handler
            file_handler = logging.FileHandler(os.path.join(log_dir, 'network.log'))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # Set level
            logger.setLevel(log_level)
        
        return logger

    def __init__(self, adjacency_matrix: np.array, params: dict,
                 origin_nodes: list, destination_nodes: list = [], 
                 demand_pattern: Callable[[int, dict], np.ndarray] = None,
                 od_flows: dict = None, pos: dict = None,
                 log_level: int = logging.INFO):
        """
        Initialize the network, with nodes and links according to the adjacency matrix
        """
        # Set up logger
        self.logger = self.setup_logger(log_level=log_level)
        
        self.adjacency_matrix = adjacency_matrix
        self.nodes = {}
        self.links = {}  # key is tuple(start_node_id, end_node_id)
        self.params = params
        self.simulation_steps = params['simulation_steps']
        self.unit_time = params['unit_time']
        self.destination_nodes = destination_nodes
        self.origin_nodes = origin_nodes
        self.pos = pos
        self.assign_flows_type = params.get('assign_flows_type', 'classic')
        self.logger.info(f"Network initialization started, assign flows type: {self.assign_flows_type}")
        
        # Initialize demand generator with passed logger
        self.demand_generator = DemandGenerator(self.simulation_steps, params, self.logger)
        
        if demand_pattern:
            self.demand_generator.register_pattern(self.params.get('custom_pattern'), demand_pattern)
            self.logger.debug(f"Custom demand pattern registered: {self.params.get('custom_pattern')}")
        
        # Initialize network structure
        self.init_nodes_and_links()
        self.logger.info(f"Network initialized with {len(self.nodes)} nodes and {len(self.links)} links")

        # Initialize managers if destination nodes are specified
        if destination_nodes:
            self.od_manager = ODManager(self.simulation_steps, logger=self.logger)
            self.od_manager.init_od_flows(origin_nodes, destination_nodes, od_flows)
        
            self.path_finder = PathFinder(self.links)
            self.path_finder.find_od_paths(od_pairs=self.od_manager.od_flows.keys(), 
                                         nodes=self.nodes)

    def _create_origin_destination(self, node: Node):
        """Create virtual links for origin/destination nodes and set demand"""
        node._create_virtual_link(node.node_id, "in", is_incoming=True, 
                                params=self.params)
        node._create_virtual_link(node.node_id, "out", is_incoming=False, 
                                params=self.params)
        
        if node.node_id in self.origin_nodes:
            # Get demand configuration for this origin
            origin_config = self.params.get('demand', {}).get(f'origin_{node.node_id}', {})
            pattern = origin_config.get('pattern', 'gaussian_peaks')
            node.demand = self.demand_generator.generate_custom(node.node_id, pattern)
        else:
            node.demand = np.zeros(self.simulation_steps)

    def _create_nodes(self, node_id: int) -> Node:
        """
        Creates a node based on its connection counts from the adjacency matrix.

        Args:
            node_id: The unique identifier of the node to create

        Returns:
            Node: An instance of appropriate Node type
        """
        incoming_count = np.sum(self.adjacency_matrix[:, node_id])
        outgoing_count = np.sum(self.adjacency_matrix[node_id, :])

        if incoming_count == 2 and outgoing_count == 2:
            if node_id in self.origin_nodes or node_id in self.destination_nodes:
                node = RegularNode(node_id=node_id)
                self._create_origin_destination(node)
            else:
                node = OneToOneNode(node_id=node_id)
        elif incoming_count == 1 and outgoing_count == 1:
            node = OneToOneNode(node_id=node_id)
            self._create_origin_destination(node)
        else:
            node = RegularNode(node_id=node_id)
            if node_id in self.origin_nodes or node_id in self.destination_nodes:
                self._create_origin_destination(node)
        return node

    def init_nodes_and_links(self):
        """Initialize nodes and links based on the adjacency matrix."""
        num_nodes = self.adjacency_matrix.shape[0]
        created_nodes = []

        for i in range(num_nodes):
            if i in created_nodes:
                node = self.nodes[i]
            else:
                node = self._create_nodes(i)
                created_nodes.append(i)
                self.nodes[i] = node

            for j in range(i+1, num_nodes):
                if self.adjacency_matrix[i, j] == 1:
                    # Create forward link (i->j)
                    if j not in created_nodes:
                        node_j = self._create_nodes(j)
                        created_nodes.append(j)
                        self.nodes[j] = node_j
                    
                    # Get link-specific parameters
                    link_params = self.params.get('links', {}).get(f'{i}_{j}', 
                                self.params.get('default_link', {}))
                    
                    # Create forward and reverse links
                    forward_link = Link(f"{i}_{j}", node, self.nodes[j], self.simulation_steps, self.unit_time, **link_params)
                    reverse_link = Link(f"{j}_{i}", self.nodes[j], node, self.simulation_steps, self.unit_time, **link_params)
                    
                    # Add links to nodes
                    node.outgoing_links.append(forward_link)
                    self.nodes[j].incoming_links.append(forward_link)
                    node.incoming_links.append(reverse_link)
                    self.nodes[j].outgoing_links.append(reverse_link)
                    
                    # Store links and set reverse links
                    self.links[(i, j)] = forward_link
                    self.links[(j, i)] = reverse_link
                    forward_link.reverse_link = reverse_link
                    reverse_link.reverse_link = forward_link

            node.init_node()

    def update_turning_fractions_per_node(self, node_ids: List[int], # function external call
                                        new_turning_fractions: np.array):
        """Update turning fractions for specified nodes"""
        for i, n in enumerate(node_ids):
            node = self.nodes[n]
            node.update_matrix_A_eq(new_turning_fractions[i])

    def update_link_states(self, time_step: int):
        """Update link states for the current time step"""
        # Update density first
        for link in self.links.values():
            link.update_link_density_flow(time_step)
        # Then update speeds
        for link in self.links.values():
            link.update_speeds(time_step)

    def network_loading(self, time_step: int):
        """Perform network loading for the current time step"""
        for node_id, node in self.nodes.items():
            if node.turning_fractions is None:
                phi = 1/(node.dest_num - 1)
                node.turning_fractions = np.ones(node.edge_num) * phi
            
            if self.destination_nodes:
                self.path_finder.calculate_node_turning_fractions(
                    time_step=time_step, 
                    od_manager=self.od_manager, 
                    node=node
                )

            if isinstance(node, OneToOneNode):
                node.assign_flows(time_step)
            else:
                if node.A_ub is None:
                    node.get_matrix_A()
                node.assign_flows(time_step, type=self.assign_flows_type)
            
        self.update_link_states(time_step)

    def visualize(self, figsize=(8, 10), node_size=200, edge_width=1,
                 show_labels=True, label_font_size=20, alpha=0.8):
        """
        Visualize the network using networkx and matplotlib.
        
        Args:
            figsize: Figure size tuple
            node_size: Size of nodes in visualization
            edge_width: Width of edges in visualization
            show_labels: Whether to show node labels
            label_font_size: Font size for labels
            alpha: Transparency level
        """
        graph = nx.DiGraph()

        # Add nodes and edges
        for node_id in self.nodes:
            graph.add_node(node_id)
        for (u, v), link in self.links.items():
            graph.add_edge(u, v, label=link.link_id)

        plt.figure(figsize=figsize)
        
        if self.pos is None:
            self.pos = nx.spring_layout(graph, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(graph, self.pos, 
                             node_size=node_size, 
                             node_color='lightblue',
                             alpha=alpha)
        
        # Separate and draw bidirectional and unidirectional edges
        bidirectional_edges = [(u, v) for (u, v) in graph.edges() 
                             if (v, u) in graph.edges()]
        unidirectional_edges = [(u, v) for (u, v) in graph.edges() 
                               if (v, u) not in graph.edges()]
        
        nx.draw_networkx_edges(graph, self.pos, 
                             edgelist=bidirectional_edges,
                             arrows=True, 
                             arrowsize=2,
                             edge_color='lightblue',
                             width=edge_width,
                             alpha=alpha,
                             connectionstyle="arc3,rad=0.2")
        
        nx.draw_networkx_edges(graph, self.pos, 
                             edgelist=unidirectional_edges,
                             arrows=True, 
                             arrowsize=2,
                             edge_color='blue',
                             width=edge_width,
                             alpha=alpha)

        if show_labels:
            nx.draw_networkx_labels(graph, self.pos, 
                                  font_size=label_font_size,
                                  alpha=alpha+0.2)

        plt.axis('off')
        plt.tight_layout()
        plt.show()