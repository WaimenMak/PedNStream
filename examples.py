# -*- coding: utf-8 -*-
# @Time    : 26/11/2024 12:10
# @Author  : mmai
# @FileName: draft
# @Software: PyCharm

# from scipy.optimize import linprog
# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt

# """
# A Link Transmission Model for the Pedestrian Traffic
# """

# def travel_cost(link, inflow, outflow):
#     """
#     Calculate the travel time of a link
#     :param link: Link object
#     :param inflow: U_i(t)
#     :param outflow: V_i(t)
#     :return: travel time of the link
#     """
#     return link.length / (link.free_flow_speed * (1 - (inflow + outflow) / link.capacity))

# def cal_travel_speed(density, v_f, k_critical, k_jam):
#     """
#     Calculate the travel speed of a link based on density.

#     Args:
#         density: Density of the link (peds/unit length).
#         v_f: Free flow speed (unit length/unit time).
#         k_critical: Critical density (peds/unit length).
#         k_jam: Jam density (peds/unit length).

#     Returns:
#         Travel speed of the link (unit length/unit time).
#     """
#     if density <= k_critical:  # Corrected condition: <=
#         return v_f  # Free flow speed/
#     elif k_critical < density:
#         # Linear decrease in speed from v_max to 0
#         return max(0, v_f * (1 - density / k_jam))


# def cal_free_flow_speed(density_i, density_j, v_f):
#     """
#     Calculate the travel speed based on densities of current and reverse links.
    
#     Args:
#         density_i: Density of the current link (k_i)
#         density_j: Density of the reverse link (k_j)
#         v_f: Free flow speed
    
#     Returns:
#         Travel speed (v = ρ * v_f, where ρ = k_i/(k_i + k_j))
#     """
#     if density_i + density_j == 0:  # Avoid division by zero
#         return v_f
    
#     rho = density_i / (density_i + density_j)
#     return rho * v_f

# def cal_travel_time(link_length, density, v_max, k_critical, k_jam):
#     """
#     Calculate the travel time of a link based on density and link length.

#     Args:
#         link_length: Length of the link (unit length).
#         density: Density of the link (vehicles/unit length).
#         v_max: Free flow speed (unit length/unit time).
#         k_critical: Critical density (vehicles/unit length).
#         k_jam: Jam density (vehicles/unit length).

#     Returns:
#         Travel time of the link (unit time). Returns infinity if speed is 0 (congestion).
#     """
#     speed = cal_travel_speed(density, v_max, k_critical, k_jam)
#     if speed == 0:
#       return float('inf')
#     return link_length / speed


# class BaseLink:
#     """Base class for all link types with common functionality"""
#     def __init__(self, link_id, start_node, end_node, simulation_steps):
#         self.link_id = link_id
#         self.start_node = start_node
#         self.end_node = end_node
        
#         # Common dynamic attributes
#         self.inflow = np.zeros(simulation_steps)
#         self.outflow = np.zeros(simulation_steps)
#         self.cumulative_inflow = np.zeros(simulation_steps)
#         self.cumulative_outflow = np.zeros(simulation_steps)

#     def update_cum_outflow(self, q_j: float, time_step: int):
#         self.outflow[time_step] = q_j
#         self.cumulative_outflow[time_step] = self.cumulative_outflow[time_step - 1] + q_j

#     def update_cum_inflow(self, q_i: float, time_step: int):
#         self.inflow[time_step] = q_i
#         self.cumulative_inflow[time_step] = self.cumulative_inflow[time_step - 1] + q_i

#     def update_speeds(self, time_step: int):
#         return

# class Link(BaseLink):
#     """Physical link with full traffic dynamics"""
#     def __init__(self, link_id, start_node, end_node, length, width, 
#                  free_flow_speed, k_critical, k_jam, unit_time, simulation_steps):
#         super().__init__(link_id, start_node, end_node, simulation_steps)
        
#         # Physical attributes
#         self.length = length
#         self.width = width
#         self.area = length * width
#         self.free_flow_speed = free_flow_speed
#         self.capacity = self.free_flow_speed * k_critical
#         self.k_jam = k_jam
#         self.k_critical = k_critical
#         self.shockwave_speed = self.capacity / (self.k_jam - self.k_critical)
#         self.current_speed = free_flow_speed
#         self.travel_time = length / free_flow_speed
#         self.unit_time = unit_time
#         self.free_flow_tau = round(self.travel_time / unit_time)

#         # Additional dynamic attributes
#         self.num_pedestrians = np.zeros(simulation_steps)
#         self.density = np.zeros(simulation_steps)
#         self.speed = np.zeros(simulation_steps)
#         self.outflow_buffer = dict()
#         self.gamma = 2e-2
#         self.receiving_flow = []
#         self.reverse_link = None

#     def update_density(self, time_step: int):
#         num_peds = self.inflow[time_step] - self.outflow[time_step]
#         self.num_pedestrians[time_step] = self.num_pedestrians[time_step - 1] + num_peds
#         self.density[time_step] = self.num_pedestrians[time_step] / self.area

#     def update_speeds(self, time_step: int):
#         """
#         Update the speed of the link based on the density
#         :param time_step:
#         :return:
#         """
#         # update speed
#         # speed = cal_travel_speed(self.density[time_step], self.free_flow_speed, k_critical=self.k_critical, k_jam=self.k_jam)
#         # Get reverse link density
#         reverse_density = 0
#         if self.reverse_link is not None:
#             reverse_density = self.reverse_link.density[time_step]

#         # Calculate new speed using density ratio formula
#         speed = cal_free_flow_speed(
#             density_i=self.density[time_step],
#             density_j=reverse_density,
#             v_f=self.free_flow_speed
#         )

#         # Update travel time and speed
#         self.speed[time_step] = speed
#         self.travel_time = self.length / speed if speed > 0 else float('inf')

#     def cal_sending_flow(self, time_step: int) -> float:
#         if self.travel_time == float('inf'):
#             return 0

#         tau = round(self.travel_time / self.unit_time)
#         if time_step - tau < 0:
#             sending_flow = 0
#         else:
#             sending_flow_boundary = self.cumulative_inflow[time_step - tau] - self.cumulative_outflow[time_step - 1]
#             sending_flow_max = self.k_critical * self.free_flow_speed * self.unit_time
#             sending_flow = min(sending_flow_boundary, sending_flow_max)

#         return sending_flow

#     def cal_receiving_flow(self, time_step: int) -> float:
#         if time_step - round(self.length/(self.shockwave_speed * self.unit_time)) < 0:
#             receiving_flow_boundary = self.k_jam * self.length
#         else:
#             receiving_flow_boundary = (self.cumulative_outflow[time_step - round(self.length/(self.shockwave_speed * self.unit_time))]
#                               + self.k_jam * self.length - self.cumulative_inflow[time_step - 1])

#         receiving_flow_max = self.k_critical * self.free_flow_speed * self.unit_time
#         receiving_flow = min(receiving_flow_boundary, receiving_flow_max)
#         # if receiving_flow < 30:
#         #     print(receiving_flow, receiving_flow_boundary, receiving_flow_max, time_step)

#         self.receiving_flow.append(receiving_flow)
#         return receiving_flow

# class Node:
#     def __init__(self, node_id):
#         self.node_id = node_id
#         self.incoming_links = []
#         self.outgoing_links = []
#         self.turning_fractions = None
#         self.mask = None
#         self.q = None
#         self.w = 1e-2
#         self.source_num = None
#         self.dest_num = None
#         self.edge_num = None
#         self.A_ub = None
#         self.A_eq = None
#         self.virtual_incoming_link = None
#         self.virtual_outgoing_link = None
#         self.M = 1e6 # for destination node, large constant for receiving flow
#         self.demand = None # for origin node

#     def _create_virtual_link(self, node_id, direction, is_incoming, params: dict):
#         """Helper method to create virtual links for origin and destination nodes"""
#         link = BaseLink(
#             link_id=f"virtual_{direction}_{node_id}",
#             start_node=self if not is_incoming else None,
#             end_node=self if is_incoming else None,
#             simulation_steps=params['simulation_steps']
#         )
#         if is_incoming:
#             self.incoming_links.append(link)
#             self.virtual_incoming_link = link
#         else:
#             self.outgoing_links.append(link)
#             self.virtual_outgoing_link = link
#         return link

#     def init_node(self):
#         """Initializes node-specific attributes based on the type."""
#         # source number is the number of incoming links
#         self.source_num = len(self.incoming_links)
#         self.dest_num = len(self.outgoing_links)
#         self.edge_num = self.dest_num * self.source_num
#         self.mask = np.ones((self.source_num, self.dest_num))
#         np.fill_diagonal(self.mask, 0)
#         self.mask = self.mask.flatten()

#         # elif isinstance(self, OriginNode):
#         #     # self.source_num = 1
#         #     self.dest_num = len(self.outgoing_links)
#         #     self.edge_num = 1 * self.dest_num

#         # elif isinstance(self, DestinationNode):
#         #     self.source_num = len(self.incoming_links)
#         #     # self.dest_num = 1
#         #     self.edge_num = self.source_num * 1

#     def update_turning_fractions(self, turning_fractions: np.array):
#         """
#         turning_fractions is a 1D array, the length is the number of edges
#         """
#         self.turning_fractions = turning_fractions
#         self.turning_fractions = self.turning_fractions * self.mask # mask the turning fractions for the edges from the same source-destination pair
                
#     def get_matrix_A(self):
#         if self.edge_num > 1:
#             row_num = self.source_num + self.dest_num
#             self.A_ub = np.zeros((row_num, self.edge_num + 2 * self.edge_num))

#             # set the constraints for the source node
#             for i in range(self.source_num):
#                 e = np.ones(self.dest_num)
#                 e[i] = 0
#                 self.A_ub[i, i * self.dest_num : (i+1) * self.dest_num] = e

#             # set the constraints for the destination node
#             for j in range(self.dest_num):
#                 for k in range(self.source_num):
#                     if k != j:
#                         self.A_ub[self.source_num + j, j + k * self.dest_num] = 1 # consider the flow on
#                     else:
#                         self.A_ub[self.source_num + j, j + k * self.dest_num] = 0

#             # turning fractions constraints
#             # if self.A_eq is not None:
#             self.update_matrix_A_eq(self.turning_fractions) # first init is equal probability

#     def update_matrix_A_eq(self, turning_fractions: np.array):
#         """
#         Update the turning fractions matrix A_eq more efficiently
        
#         Args:
#             turning_fractions: Array of turning fraction values
#         """
#         self.turning_fractions = turning_fractions
#         assert len(turning_fractions) == self.edge_num
        
#         # Initialize A_eq matrix with zeros
#         self.A_eq = np.zeros((self.edge_num - self.source_num, self.edge_num + 2 * self.edge_num))
        
#         # Use vectorized operations where possible
#         non_zero_flows = turning_fractions != 0
#         indices = np.where(non_zero_flows)[0]
        
#         for i, l in enumerate(indices):
#             source_idx = l // self.dest_num
#             start_ind = source_idx * self.dest_num
            
#             # Set turning fractions for all destinations from this source
#             self.A_eq[i, start_ind:start_ind + self.dest_num] = turning_fractions[l]
#             # the penalty term for the edge from the same source-destination pair should be 0
#             self.A_eq[i, l] = turning_fractions[l] - 1 # the lth column is phi - 1
#             self.A_eq[i, start_ind + source_idx] = 0 
#             self.A_eq[i, self.edge_num + l * 2 : self.edge_num + (l+1) * 2] = np.array([1, -1])  # the penalty term

#     def update_links(self, time_step):
#         """Update the upstream link's downstream cumulative outflow
#         q -->> [S1, S2, R1, R2, R3], already sum up the flows from/to the same link
#         """
#         assert self.q is not None
#         # whether length of q is number of edges
#         assert len(self.q) == self.edge_num

#         for l, link in enumerate(self.incoming_links):
#             inflow = self.q[l]
#             link.update_cum_outflow(inflow, time_step)
#             # link.update_speeds(time_step)

#         for m, link in enumerate(self.outgoing_links):
#             outflow = self.q[self.source_num + m]
#             link.update_cum_inflow(outflow, time_step)
#             # link.update_speeds(time_step)
    
#     def solve(self, s, r):
#         return

#     def assign_flows(self, time_step: int):
#         s = np.zeros(self.source_num)
#         r = np.zeros(self.dest_num)
        
#         # Calculate sending flows
#         for i, l in enumerate(self.incoming_links):
#             if hasattr(self, 'virtual_incoming_link') and l == self.virtual_incoming_link:
#                 s[i] = self.demand[time_step]
#             else:
#                 s[i] = l.cal_sending_flow(time_step)
        
#         # Calculate receiving flows
#         for j, l in enumerate(self.outgoing_links):
#             if hasattr(self, 'virtual_outgoing_link') and l == self.virtual_outgoing_link:
#                 r[j] = self.M
#             else:
#                 r[j] = l.cal_receiving_flow(time_step)
        
#         self.solve(s, r)
#         self.update_links(time_step)

# # class OriginNode(Node):
# #     def __init__(self, node_id):
# #         super().__init__(node_id)

# #     def assign_flows(self, time_step):
# #         assert self.demand is not None
# #         s = np.array([self.demand[time_step]]) # for origin node, the demand is the outflow of its upstream link already.
# #         r = np.zeros(self.dest_num)
# #         for j, l in enumerate(self.outgoing_links):
# #             r[j] = l.cal_receiving_flow(time_step)

# #         if self.edge_num == 1: # when the node has only one outgoing link
# #             self.q_ij = np.array([np.min([s[0], r[0]])]) # this case s and r are scalars
# #             self.update_links(time_step)
# #             return

# #         # solve the linear programming problem
# #         w = self.w * np.ones(2 * self.edge_num)
# #         c = -1 * np.ones(self.edge_num)
# #         c = np.concatenate((c, w))
# #         b_ub = np.concatenate((s, r))
# #         res = linprog(c, A_ub=self.A_ub, A_eq=self.A_eq, b_ub=b_ub, b_eq=np.zeros(self.edge_num))

# #         if res.success:
# #             # flows = res.x[:self.edge_num]
# #             self.q_ij = np.floor(self.A_ub @ res.x)
# #             # total_flow = -result.fun
# #             self.update_links(time_step)
# #         return

# # class DestinationNode(Node):
# #     def __init__(self, node_id):
# #         super().__init__(node_id)

# #     def assign_flows(self, time_step):
# #         s = np.zeros(self.source_num)
# #         for i, l in enumerate(self.incoming_links):
# #             s[i] = l.cal_sending_flow(time_step)
# #         self.q_ij = s
# #         self.update_links(time_step)
# #         return
    


# class OneToOneNode(Node):
#     def __init__(self, node_id):
#         super().__init__(node_id)

#     def solve(self, s, r):
#         """
#         q = [S0, S1, R0, R1], S1 and R1 are virtual links
#         """
#         self.q = np.array([np.min([s[0],r[1]]), np.min([s[1],r[0]]),
#                             np.min([s[1],r[0]]), np.min([s[0],r[1]])])
#         return


# class RegularNode(Node):
#     def __init__(self, node_id):
#         super().__init__(node_id)

#     def solve(self, s, r):
#         # solve the linear programming problem
#         w = self.w * np.ones(2 * self.edge_num)
#         c = -1 * np.ones(self.edge_num)
#         c = np.concatenate((c, w))
#         b_ub = np.concatenate((s, r))

#         if self.A_eq is not None:
#             res = linprog(c, A_ub=self.A_ub, A_eq=self.A_eq, b_ub=b_ub, b_eq=np.zeros(self.edge_num))
#         else:
#             res = linprog(c, A_ub=self.A_ub, b_ub=b_ub)

#         if res.success:
#             flows = self.A_ub @ res.x
#             flows[np.arange(min(self.source_num, self.dest_num)) * (self.dest_num + 1)] = 0
#             self.q = np.floor(flows)
#         return


# class Network:
#     def __init__(self, adjacency_matrix: np.array, link_params: dict):
#         """
#         Initialize the network, with nodes and links according to the adjacency matrix
#         """
#         self.adjacency_matrix = adjacency_matrix
#         self.nodes = []
#         self.links = {}          #use dict to store link, key is tuple(start_node_id, end_node_id)
#         self.link_params = link_params
#         self.simulation_steps = link_params['simulation_steps']
#         self.init_nodes_and_links()

#     # TODO: update the turning fractions
#     # def update_turning_fractions(self, new_turning_fractions):
#     #     for node in self.nodes:
#     #         if node.source_num > 0 and node.dest_num > 0:
#     #             if node.turning_fractions is None:
#     #                 phi = 1/node.source_num
#     #                 node.turning_fractions = np.ones(node.edge_num) * phi
#     #             else:
#     #                 node.turning_fractions = new_turning_fractions

#     def init_nodes_and_links(self, lam=10):
#         """
#         Initialize nodes and links based on the adjacency matrix.
#         For a complete graph, excluding self-loops.
#         """
#         num_nodes = self.adjacency_matrix.shape[0]

#         # Create nodes
#         for i in range(num_nodes):
#             self.nodes.append(Node(node_id=i))

#         # Create links (excluding self-loops)
#         for i in range(num_nodes):
#             for j in range(num_nodes):
#                 # Skip self-loops
#                 if i == j:
#                     continue
#                 if i != j:  
#                     link_id = f"{i}_{j}"
#                     link = Link(link_id, self.nodes[i], self.nodes[j], **self.link_params)
#                     self.links[(i, j)] = link
#                     self.nodes[i].outgoing_links.append(link)
#                     self.nodes[j].incoming_links.append(link)
                    
#                     # Connect reverse links
#                     if (j, i) in self.links:
#                         reverse_link = self.links[(j, i)]
#                         link.reverse_link = reverse_link
#                         reverse_link.reverse_link = link

#         # assign the OD node manually
#         od_nodes = [0, 1]
#         origin_nodes = [0, 1]
#         for node_id in od_nodes:
#             node = self.nodes[node_id]
#             node._create_virtual_link(node_id, "in", is_incoming=True, params=self.link_params)
#             node._create_virtual_link(node_id, "out", is_incoming=False, params=self.link_params)
#             if node_id in origin_nodes:
#                 node.demand = np.random.poisson(lam=10, size=self.simulation_steps)
#             else:
#                 node.demand = np.zeros(self.simulation_steps) # no demand for destination nodes
        

#         for node in self.nodes:
#             if len(node.outgoing_links) == 2 and len(node.incoming_links) == 2:
#                 node.__class__ = OneToOneNode
#             else:
#                 # Regular node
#                 node.__class__ = RegularNode
#             node.init_node() #Initialize the node after determine the type

#     def update_transition_probability(self, new_turning_fractions_matrix):
#         pass

#     def update_link_states(self, time_step: int):
#         """
#         Cannot be in the same loop as network_loading, 
#         because the density and speed are updated based on the previous time step
#         """
#         # update density
#         for link in self.links.values():
#             link.update_density(time_step)
#         # update speed
#         for link in self.links.values():
#             link.update_speeds(time_step)

#     def network_loading(self, time_step: int):
#         for node in self.nodes:
#             if node.turning_fractions is None:
#                 phi = 1/(node.dest_num - 1) # the number of edges from the different source-destination pair
#                 turning_fractions = np.ones(node.edge_num) * phi
#                 node.update_turning_fractions(turning_fractions)
#             else:
#                 # TODO: update the turning fractions
#                 # node.update_turning_fractions(new_turning_fractions)
#                 pass

#             if isinstance(node, OneToOneNode):
#                 node.assign_flows(time_step)
#             else:
#                 node.get_matrix_A()  # only for RegularNode and OriginNode with multiple outgoing links
#                 node.assign_flows(time_step)
            
#         self.update_link_states(time_step)

#     def visualize(self):
#         """Visualizes the network using networkx and matplotlib."""
#         graph = nx.DiGraph()

#         # Add nodes
#         for node in self.nodes:
#             graph.add_node(node.node_id)

#         # Add edges (links) with labels
#         for (u, v), link in self.links.items():
#             graph.add_edge(u, v, label=link.link_id)

#         # Drawing options (adjust as needed)
#         pos = nx.spring_layout(graph, seed=42)  # For consistent layout
#         node_options = {"node_size": 700, "node_color": "skyblue", "font_size": 15, "font_weight": "bold"}
#         edge_options = {"arrowstyle": "-|>", "arrowsize": 20}  # Add arrows for directed graph

#         # Draw nodes and edges
#         nx.draw(graph, pos, with_labels=True, **node_options)
#         nx.draw_networkx_edges(graph, pos, **edge_options) # Draw edges with arrows

#         # Draw edge labels
#         edge_labels = nx.get_edge_attributes(graph, 'label')
#         nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10)

#         plt.tight_layout() # Prevents labels from being cut off
#         plt.show()


from src.LTM.network import Network
import numpy as np
import matplotlib.pyplot as plt
from src.handlers.output_handler import OutputHandler

if __name__ == "__main__":
    # Network configuration
    # adj = np.array([[0, 1],
    #                 [1, 0]])
    
    # adj = np.array([[0, 1, 1, 1],
    #                 [1, 0, 1, 1],
    #                 [1, 1, 0, 1],
    #                 [1, 1, 1, 0]])

    adj = np.array([[0, 1, 1, 1, 0, 0],
                    [1, 0, 1, 1, 0, 0],
                    [1, 1, 0, 1, 0, 0],
                    [1, 1, 1, 0, 1, 0],
                    [0, 0, 0, 1, 0, 1],
                    [0, 0, 0, 0, 1, 0]])


    params = {
        'length': 100,
        'width': 1,
        'free_flow_speed': 1.5,
        'k_critical': 2,
        'k_jam': 10,
        'unit_time': 10,
        'peak_lambda': 10,
        'base_lambda': 5,
        'simulation_steps': 200,
    }

    # Initialize and run simulation
    network_env = Network(adj, params, od_nodes=[5, 4], origin_nodes=[5])
    network_env.visualize()

    # Run simulation
    for t in range(1, params['simulation_steps']):
        network_env.network_loading(t)
    
# Plot inflow and outflow
plt.figure(1)
path = [(5, 4), (4, 3), (3, 2), (2, 1), (1, 0)]
for link_id in path:
    plt.plot(network_env.links[link_id].inflow, label=f'inflow{link_id}')
    plt.legend()
plt.show()

# save the network state
output_handler = OutputHandler(base_dir="outputs")
output_handler.save_network_state(network_env)


# # Plot density and speed
# plt.figure(2)
# plt.plot(network.links[link_id].density, label='density')
# plt.plot(network.links[link_id].speed, label='speed')
# plt.legend()
# plt.show()

# # Plot cumulative flows
# plt.figure(3)
# plt.plot(network.links[link_id].cumulative_inflow, label='cumulative_inflow')
# plt.plot(network.links[link_id].cumulative_outflow, label='cumulative_outflow')
# plt.legend()
# plt.show()