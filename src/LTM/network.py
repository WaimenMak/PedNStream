import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from .node import Node, OneToOneNode, RegularNode
from .link import Link
from .od_manager import ODManager
from .path_finder import PathFinder


"""
A Link Transmission Model for the Pedestrian Traffic
"""

class Network:
    def __init__(self, adjacency_matrix: np.array, link_params: dict,
                 origin_nodes: list, destination_nodes: list = [], od_flows: dict = None, pos: dict = None):
        """
        Initialize the network, with nodes and links according to the adjacency matrix
        """
        self.adjacency_matrix = adjacency_matrix
        self.nodes = {}
        self.links = {}          #use dict to store link, key is tuple(start_node_id, end_node_id)
        self.link_params = link_params
        self.simulation_steps = link_params['simulation_steps']
        self.unit_time = link_params['unit_time']
        self.destination_nodes = destination_nodes
        self.origin_nodes = origin_nodes
        self.init_nodes_and_links()
        self.pos = pos

        # Initialize managers
        if destination_nodes:
            self.od_manager = ODManager(link_params['simulation_steps'])
            self.od_manager.init_od_flows(origin_nodes, destination_nodes, od_flows)
        
            self.path_finder = PathFinder(self.links)
            self.path_finder.find_od_paths(od_pairs=self.od_manager.od_flows.keys(), nodes=self.nodes)

    # TODO: update the turning fractions
    # def update_turning_fractions(self, new_turning_fractions):
    #     for node in self.nodes:
    #         if node.source_num > 0 and node.dest_num > 0:
    #             if node.turning_fractions is None:
    #                 phi = 1/node.source_num
    #                 node.turning_fractions = np.ones(node.edge_num) * phi
    #             else:
    #                 node.turning_fractions = new_turning_fractions

    def update_turning_fractions_per_node(self, node_ids: list, new_turning_fractions: np.array):
        for i, n in enumerate(node_ids):
            node = self.nodes[n]
            node.update_matrix_A_eq(new_turning_fractions[i])

    def _generate_time_varying_demand(self, simulation_steps, params):
        """Generate a time-varying demand pattern with morning and evening peaks"""
        time = np.arange(simulation_steps)
        peak_lambda = params.get('peak_lambda', 10)  # default value if not specified
        base_lambda = params.get('base_lambda', 5)   # default value if not specified

        t = 300
        morning_peak = peak_lambda * np.exp(-(time - t/4)**2 / (2 * (t/20)**2))
        evening_peak = peak_lambda * np.exp(-(time - 3*t/4)**2 / (2 * (t/20)**2))
        lambda_t = base_lambda + morning_peak + evening_peak
        seed = params.get('seed', 42)
        np.random.seed(seed)
        # after a certian time, the demand is decreasing
        demand = np.random.poisson(lam=lambda_t)
        # create a spike at from t == 400 to t == 410
        # if t == 400:
        # demand[0:10] = 100
        demand[250:260] = 100
        # demand[time > 10] = 0
        # demand[time > 300] = 0
        # demand = np.ones(simulation_steps) * 1
        demand[time > 400] = 0
        return demand

    # def init_nodes_and_links(self, origin_nodes: list):
    #     """
    #     Initialize nodes and links based on the adjacency matrix.
    #     For a complete graph, excluding self-loops.
    #     """
    #     num_nodes = self.adjacency_matrix.shape[0]

    #     # Create nodes
    #     for i in range(num_nodes):
    #         self.nodes.append(Node(node_id=i))

    #     # Create links (excluding self-loops)
    #     for i in range(num_nodes):
    #         for j in range(num_nodes):
    #             # Skip self-loops
    #             if i != j and self.adjacency_matrix[i, j] == 1:
    #                 link_id = f"{i}_{j}"
    #                 link = Link(link_id, self.nodes[i], self.nodes[j], **self.link_params)
    #                 self.links[(i, j)] = link
    #                 self.nodes[i].outgoing_links.append(link)
    #                 self.nodes[j].incoming_links.append(link)
                    
    #                 # Connect reverse links
    #                 if (j, i) in self.links:
    #                     reverse_link = self.links[(j, i)]
    #                     link.reverse_link = reverse_link
    #                     reverse_link.reverse_link = link

    #     # assign the OD node manually, for origin node, it has a virtual link in the in/out links list, but not present in the network's links
    #     for node_id in origin_nodes:
    #         node = self.nodes[node_id]
    #         node._create_virtual_link(node_id, "in", is_incoming=True, params=self.link_params)
    #         node._create_virtual_link(node_id, "out", is_incoming=False, params=self.link_params)
    #         if node_id in origin_nodes:
    #             node.demand = self._generate_time_varying_demand(self.simulation_steps, self.link_params)
        

    #     for node in self.nodes:
    #         has_virtual_links = node.virtual_incoming_link is not None
    #         is_dead_end = (len(node.incoming_links) == 1 or len(node.outgoing_links) == 1) and not has_virtual_links
            
    #         if is_dead_end: # if it is dead end treat it as destination node
    #             # print(f"Dead End Detected at node {node.node_id}")
    #             # raise AssertionError(f"No Dead End, Please! Detected at node {node.node_id}")
    #             node._create_virtual_link(node.node_id, "in", is_incoming=True, params=self.link_params)
    #             node._create_virtual_link(node.node_id, "out", is_incoming=False, params=self.link_params)
    #             node.demand = np.zeros(self.simulation_steps) # no demand for destination nodes

    #         if len(node.outgoing_links) == 2 and len(node.incoming_links) == 2:
    #             node.__class__ = OneToOneNode
    #         # elif len(node.outgoing_links) == 0 or len(node.incoming_links) == 0:
    #         #     #remove the node from the list
    #         #     self.nodes.remove(node)
    #         else:
    #             # Regular node
    #             node.__class__ = RegularNode
    #         node.init_node() #Initialize the node after determine the type
    def _create_origin_destination(self, node):
        node._create_virtual_link(node.node_id, "in", is_incoming=True, params=self.link_params)
        node._create_virtual_link(node.node_id, "out", is_incoming=False, params=self.link_params)
        if node.node_id in self.origin_nodes:
            node.demand = self._generate_time_varying_demand(self.simulation_steps, self.link_params)
        else:
            node.demand = np.zeros(self.simulation_steps)


    def _create_nodes(self, node_id: int):
        """
        Creates a node based on its connection counts from the adjacency matrix.

        Args:
            node_id (int): The unique identifier of the node to create

        Returns:
            Node: An instance of Node (either RegularNode, OneToOneNode, or a node
                  with virtual links if it's an origin/destination)
        """
        incoming_count = np.sum(self.adjacency_matrix[:, node_id])  # Count connections coming into the node
        outgoing_count = np.sum(self.adjacency_matrix[node_id, :])  # Count connections going out from the node

        # Determine node type based on connection counts
        if incoming_count == 2 and outgoing_count == 2:  # (2 edges)
        # If node has exactly 2 incoming and 2 outgoing connections
            if node_id in self.origin_nodes or node_id in self.destination_nodes:
            # If node is designated as origin or destination, create RegularNode
                node = RegularNode(node_id=node_id)
                self._create_origin_destination(node)
            else:
            # Otherwise, create OneToOneNode for regular internal nodes
                node = OneToOneNode(node_id=node_id)
            
        elif incoming_count == 1 and outgoing_count == 1: # (dead end)
            node = OneToOneNode(node_id=node_id)
            self._create_origin_destination(node)
        else:   # (regular node)
            node = RegularNode(node_id=node_id)
            if node_id in self.origin_nodes or node_id in self.destination_nodes:
                self._create_origin_destination(node)
        return node

    def init_nodes_and_links(self):
        """
        Initialize nodes and links based on the adjacency matrix.
        """
        num_nodes = self.adjacency_matrix.shape[0]
        created_nodes = []

        # Create nodes with proper type from the beginning
        for i in range(num_nodes):
            if i in created_nodes:
                node = self.nodes[i]
            else:
                node = self._create_nodes(i)
                created_nodes.append(i)
                self.nodes[i] = node
            #add outgoing and incoming links of current node
            for j in range(i+1, num_nodes):
                if self.adjacency_matrix[i, j] == 1: # symmetric adjacency matrix
                    link_id = f"{i}_{j}"
                    if j in created_nodes:
                        link = Link(link_id, node, self.nodes[j], **self.link_params)
                    else:
                        # create node j
                        node_j = self._create_nodes(j)
                        created_nodes.append(j)
                        self.nodes[j] = node_j
                        link = Link(link_id, node, node_j, **self.link_params)
                    node.outgoing_links.append(link)
                    self.nodes[j].incoming_links.append(link)
                    self.links[(i, j)] = link
                # if self.adjacency_matrix[j, i] == 1:
                    link_id = f"{j}_{i}"
                    link = Link(link_id, self.nodes[j], node, **self.link_params)
                    node.incoming_links.append(link)
                    self.nodes[j].outgoing_links.append(link)
                    self.links[(j, i)] = link

                    # reverse links
                    self.links[(j, i)].reverse_link = self.links[(i, j)]
                    self.links[(i, j)].reverse_link = self.links[(j, i)]

            node.init_node()


    def update_transition_probability(self, new_turning_fractions_matrix):
        pass

    def update_link_states(self, time_step: int):
        """
        Cannot be in the same loop as network_loading, 
        because the density and speed are updated based on the previous time step,
        so we have to update the density first and then update the speed
        """
        # update density
        for link in self.links.values():
            link.update_link_density_flow(time_step)
        # update speed
        for link in self.links.values():
            link.update_speeds(time_step)

    def network_loading(self, time_step: int):
        for node_id, node in self.nodes.items():
            if node.turning_fractions is None:
                phi = 1/(node.dest_num - 1) # the number of edges from the different source-destination pair
                turning_fractions = np.ones(node.edge_num) * phi
                # node.update_turning_fractions(turning_fractions)
                node.turning_fractions = turning_fractions
            else:
                # TODO: update the turning fractions
                # node.update_turning_fractions(new_turning_fractions)
                pass

            if self.destination_nodes:
                # cal turning fractions for nodes in paths
                self.path_finder.calculate_node_turning_fractions(time_step=time_step, od_manager=self.od_manager, node=node)

            if isinstance(node, OneToOneNode):
                node.assign_flows(time_step)
            else:
                if node.A_ub is None:
                    node.get_matrix_A() # only for RegularNode and OriginNode with multiple outgoing links
                node.assign_flows(time_step)
            
        self.update_link_states(time_step)

    def visualize(self, figsize=(15, 15), node_size=30, edge_width=1, 
                 show_labels=False, label_font_size=6, alpha=0.8):
        """Visualizes the network using networkx and matplotlib."""
        graph = nx.DiGraph()

        # Add nodes and edges as before
        for node_id, node in self.nodes.items():
            graph.add_node(node_id)
        for (u, v), link in self.links.items():
            graph.add_edge(u, v, label=link.link_id)

        # Create figure with specified size
        plt.figure(figsize=figsize)
        
        # Drawing options
        if self.pos is None:
            self.pos = nx.spring_layout(graph, k=1, iterations=50)  # k=1 increases spacing
        
        # Draw nodes with transparency
        nx.draw_networkx_nodes(graph, self.pos, 
                             node_size=node_size, 
                             node_color='blue',
                             alpha=alpha)
        
        # Draw edges with different styles for bidirectional vs unidirectional
        # Separate bidirectional and unidirectional edges
        bidirectional_edges = [(u, v) for (u, v) in graph.edges() 
                             if (v, u) in graph.edges()]
        unidirectional_edges = [(u, v) for (u, v) in graph.edges() 
                               if (v, u) not in graph.edges()]
        
        # Draw bidirectional edges
        nx.draw_networkx_edges(graph, self.pos, 
                             edgelist=bidirectional_edges,
                             arrows=True, 
                             arrowsize=2,
                             edge_color='lightblue',
                             width=edge_width,
                             alpha=alpha,
                             connectionstyle=f"arc3,rad=0.2")
        
        # Draw unidirectional edges
        nx.draw_networkx_edges(graph, self.pos, 
                             edgelist=unidirectional_edges,
                             arrows=True, 
                             arrowsize=2,
                             edge_color='blue',
                             width=edge_width,
                             alpha=alpha)

        # Optionally draw labels
        if show_labels:
            nx.draw_networkx_labels(graph, self.pos, 
                                  font_size=label_font_size,
                                  alpha=alpha+0.2)  # Labels slightly more visible than edges

        # plt.title("Network Visualization", pad=20)
        plt.axis('off')  # Turn off axis
        plt.tight_layout()
        plt.show()