import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from .node import Node, OneToOneNode, RegularNode
from .link import Link


"""
A Link Transmission Model for the Pedestrian Traffic
"""

class Network:
    def __init__(self, adjacency_matrix: np.array, link_params: dict,
                 od_nodes: list, origin_nodes: list):
        """
        Initialize the network, with nodes and links according to the adjacency matrix
        """
        self.adjacency_matrix = adjacency_matrix
        self.nodes = []
        self.links = {}          #use dict to store link, key is tuple(start_node_id, end_node_id)
        self.link_params = link_params
        self.simulation_steps = link_params['simulation_steps']
        self.unit_time = link_params['unit_time']
        self.init_nodes_and_links(od_nodes, origin_nodes)

    # TODO: update the turning fractions
    # def update_turning_fractions(self, new_turning_fractions):
    #     for node in self.nodes:
    #         if node.source_num > 0 and node.dest_num > 0:
    #             if node.turning_fractions is None:
    #                 phi = 1/node.source_num
    #                 node.turning_fractions = np.ones(node.edge_num) * phi
    #             else:
    #                 node.turning_fractions = new_turning_fractions

    def _generate_time_varying_demand(self, simulation_steps, params):
        """Generate a time-varying demand pattern with morning and evening peaks"""
        time = np.arange(simulation_steps)
        peak_lambda = params.get('peak_lambda', 10)  # default value if not specified
        base_lambda = params.get('base_lambda', 5)   # default value if not specified
        
        morning_peak = peak_lambda * np.exp(-(time - simulation_steps/4)**2 / (2 * (simulation_steps/20)**2))
        evening_peak = peak_lambda * np.exp(-(time - 3*simulation_steps/4)**2 / (2 * (simulation_steps/20)**2))
        lambda_t = base_lambda + morning_peak + evening_peak
        seed = params.get('seed', 42)
        np.random.seed(seed)
        demand = np.random.poisson(lam=lambda_t)
        
        return demand

    def init_nodes_and_links(self, od_nodes: list, origin_nodes: list):
        """
        Initialize nodes and links based on the adjacency matrix.
        For a complete graph, excluding self-loops.
        """
        num_nodes = self.adjacency_matrix.shape[0]

        # Create nodes
        for i in range(num_nodes):
            self.nodes.append(Node(node_id=i))

        # Create links (excluding self-loops)
        for i in range(num_nodes):
            for j in range(num_nodes):
                # Skip self-loops
                if i != j and self.adjacency_matrix[i, j] == 1:
                    link_id = f"{i}_{j}"
                    link = Link(link_id, self.nodes[i], self.nodes[j], **self.link_params)
                    self.links[(i, j)] = link
                    self.nodes[i].outgoing_links.append(link)
                    self.nodes[j].incoming_links.append(link)
                    
                    # Connect reverse links
                    if (j, i) in self.links:
                        reverse_link = self.links[(j, i)]
                        link.reverse_link = reverse_link
                        reverse_link.reverse_link = link

        # assign the OD node manually
        for node_id in od_nodes:
            node = self.nodes[node_id]
            node._create_virtual_link(node_id, "in", is_incoming=True, params=self.link_params)
            node._create_virtual_link(node_id, "out", is_incoming=False, params=self.link_params)
            if node_id in origin_nodes:
                node.demand = self._generate_time_varying_demand(self.simulation_steps, self.link_params)
            else:
                node.demand = np.zeros(self.simulation_steps) # no demand for destination nodes
        

        for node in self.nodes:
            has_virtual_links = node.virtual_incoming_link is not None
            is_dead_end = (len(node.incoming_links) == 1 or len(node.outgoing_links) == 1) and not has_virtual_links
            
            if is_dead_end:
                raise AssertionError(f"No Dead End, Please! Detected at node {node.node_id}")

            if len(node.outgoing_links) == 2 and len(node.incoming_links) == 2:
                node.__class__ = OneToOneNode
            else:
                # Regular node
                node.__class__ = RegularNode
            node.init_node() #Initialize the node after determine the type

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
        for node in self.nodes:
            if node.turning_fractions is None:
                phi = 1/(node.dest_num - 1) # the number of edges from the different source-destination pair
                turning_fractions = np.ones(node.edge_num) * phi
                node.update_turning_fractions(turning_fractions)
            else:
                # TODO: update the turning fractions
                # node.update_turning_fractions(new_turning_fractions)
                pass

            if isinstance(node, OneToOneNode):
                node.assign_flows(time_step)
            else:
                node.get_matrix_A()  # only for RegularNode and OriginNode with multiple outgoing links
                node.assign_flows(time_step)
            
        self.update_link_states(time_step)

    def visualize(self):
        """Visualizes the network using networkx and matplotlib."""
        graph = nx.DiGraph()

        # Add nodes
        for node in self.nodes:
            graph.add_node(node.node_id)

        # Add edges (links) with labels
        for (u, v), link in self.links.items():
            graph.add_edge(u, v, label=link.link_id)

        # Drawing options (adjust as needed)
        pos = nx.spring_layout(graph, seed=42)  # For consistent layout
        node_options = {"node_size": 700, "node_color": "skyblue"}
        
        # Draw nodes and labels separately
        nx.draw_networkx_nodes(graph, pos, **node_options)
        nx.draw_networkx_labels(graph, pos, font_size=15, font_weight="bold")

        # Draw edges with offset for bidirectional edges
        for (u, v) in graph.edges():
            # If edge is bidirectional, add offset
            if (v, u) in graph.edges():
                # Draw edge with positive offset
                nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)], 
                                     arrows=True, arrowsize=20,
                                     edge_color='b',
                                     width=1,
                                     connectionstyle=f"arc3,rad=0.2")
            else:
                # Draw normal edge
                nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)], 
                                     arrows=True, arrowsize=20)

        # Draw edge labels with adjusted positions
        edge_labels = nx.get_edge_attributes(graph, 'label')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, 
                                    font_size=10, label_pos=0.3)

        plt.tight_layout()
        plt.show()