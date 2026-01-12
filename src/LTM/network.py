import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from .node import Node, OneToOneNode, RegularNode
from .link import Link, Separator
# from .link_org import Link, Separator
# from .link_bi import Link, Separator
from .od_manager import ODManager, DemandGenerator
from .path_finder import PathFinder
from typing import Callable, List
import logging
import os
from pathlib import Path

"""
A Link Transmission Model for the Pedestrian Traffic
"""

class Network:
    @staticmethod
    def setup_logger(log_level=logging.INFO, log_dir=None):
        """Set up and configure logger"""
        if log_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            log_dir = project_root / "outputs" / "logs"
        else:
            log_dir = Path(log_dir)

        # Create logs directory if it doesn't exist
        log_dir.mkdir(parents=True, exist_ok=True)
        
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
            file_handler = logging.FileHandler(log_dir / 'network.log')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # Set level
            logger.setLevel(log_level)
        
        return logger

    def __init__(self, adjacency_matrix: np.array, params: dict,
                 origin_nodes: list, destination_nodes: list = [], 
                 demand_pattern: List[Callable[[int, dict], np.ndarray]] = None,
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
        self.params = params # already contains default link parameters, simulation steps, unit time, link specific params and demand params
        self.simulation_steps = params['simulation_steps']
        self.unit_time = params['unit_time']
        self.destination_nodes = destination_nodes
        self.origin_nodes = origin_nodes
        self.path_finder = None
        self.pos = pos
        self.assign_flows_type = params.get('assign_flows_type', 'classic')
        self.logger.info(f"Network initialization started, assign flows type: {self.assign_flows_type}")
        
        # Initialize demand generator with passed logger
        self.demand_generator = DemandGenerator(self.simulation_steps, params, self.logger)
        
        if demand_pattern:
            # self.demand_generator.register_pattern(self.params.get('custom_pattern'), demand_pattern)
            for func in demand_pattern:
                self.demand_generator.register_pattern(func.__name__, func)
                self.logger.info(f"Custom demand pattern registered: {func.__name__}")
        
# Controller configuration
        controller_config = params.get('controllers', {})
        self.controller_enabled = controller_config.get('enabled', False)
        self.controller_nodes = controller_config.get('nodes', set())  # all nodes related to controllers
        self.controller_nodes = set(map(int, self.controller_nodes))
        self.controller_gaters = self.controller_nodes.copy() # only the intersection nodes with gater control
        self.controller_links = controller_config.get('links', []) # link is a string like '1-2'
        for link in self.controller_links: # expand the nodes related to controllers, so controller nodes are all nodes related to the controllers
            start_node, end_node = link.split('-')
            self.controller_nodes.add(int(start_node))
            self.controller_nodes.add(int(end_node))
        self.logger.info(f"Controller configuration: enabled: {self.controller_enabled}, nodes: {self.controller_nodes}, links: {self.controller_links}")

        # Initialize network structure
        self.init_nodes_and_links()
        if self.logger and self.verbose:
            self.logger.info(f"Network initialized with {len(self.nodes)} nodes and {len(self.links)} links")

        # Initialize managers if destination nodes are specified
        if destination_nodes:
            self.od_manager = ODManager(self.simulation_steps, logger=self.logger if self.verbose else None)
            self.od_manager.init_od_flows(origin_nodes, destination_nodes, od_flows)
        
            self.path_finder = PathFinder(self.links, params=self.params, controller_nodes=self.controller_nodes, controller_links=self.controller_links, logger=self.logger if self.verbose else None)  # Pass params and logger here
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
            pattern = origin_config.get('pattern', 'gaussian_peaks')   # get the pattern name of the node in the yaml file
            node.demand = self.demand_generator.generate_custom(node.node_id, pattern)
            # log the total demand of the origin node
            if self.logger and self.verbose:
                self.logger.info(f"Total demand of origin node {node.node_id}: {np.sum(node.demand)}")
        else:
            node.demand = np.zeros(self.simulation_steps) # destination nodes have no demand

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

    def _get_link_params(self, i: int, j: int) -> dict:
        """
        Get link-specific parameters, checking both direction keys.
        
        Args:
            i: First node
            j: Second node
            
        Returns:
            dict: Link parameters (same for both directions)
        """
        links_config = self.params.get('links', {})
        default_params = self.params.get('default_link', {})
        
        # Check both possible keys: i_j and j_i
        forward_key = f'{i}_{j}'
        reverse_key = f'{j}_{i}'
        
        if forward_key in links_config:
            return {**default_params, **links_config[forward_key]}
        elif reverse_key in links_config:
            return {**default_params, **links_config[reverse_key]}
        else:
            return default_params

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
                    
                    # Get shared parameters for both directions
                    link_params = self._get_link_params(i, j)
                    if f"{i}-{j}" in self.controller_links or f"{j}-{i}" in self.controller_links:
                        link_type = 'separator'
                    else:
                        link_type = link_params.get('controller_type', 'gate')
                    
                    # Create both links with identical parameters
                    if link_type == 'separator':
                        forward_link = Separator(f"{i}_{j}", node, self.nodes[j], 
                                               self.simulation_steps, self.unit_time, **link_params)
                        reverse_link = Separator(f"{j}_{i}", self.nodes[j], node, 
                                               self.simulation_steps, self.unit_time, **link_params)
                    elif link_type == 'gate':
                        forward_link = Link(f"{i}_{j}", node, self.nodes[j], 
                                          self.simulation_steps, self.unit_time, **link_params)
                        reverse_link = Link(f"{j}_{i}", self.nodes[j], node, 
                                          self.simulation_steps, self.unit_time, **link_params)
                    else:
                        raise ValueError(f"Invalid controller type: {link_type}")
                    
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
        """Perform network loading for the current time step, time_step starts from 1"""
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