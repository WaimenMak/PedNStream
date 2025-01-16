# -*- coding: utf-8 -*-
# @Time    : 26/11/2024 12:10
# @Author  : mmai
# @FileName: draft
# @Software: PyCharm

from scipy.optimize import linprog
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

"""
A Link Transmission Model for the Pedestrian Traffic
"""

def travel_cost(link, inflow, outflow):
    """
    Calculate the travel time of a link
    :param link: Link object
    :param inflow: U_i(t)
    :param outflow: V_i(t)
    :return: travel time of the link
    """
    return link.length / (link.free_flow_speed * (1 - (inflow + outflow) / link.capacity))

def cal_travel_speed(density, v_f, k_critical, k_jam):
    """
    Calculate the travel speed of a link based on density.

    Args:
        density: Density of the link (vehicles/unit length).
        v_f: Free flow speed (unit length/unit time).
        k_critical: Critical density (vehicles/unit length).
        k_jam: Jam density (vehicles/unit length).

    Returns:
        Travel speed of the link (unit length/unit time).
    """
    if density <= k_critical:  # Corrected condition: <=
        return v_f  # Free flow speed
    elif k_critical < density:
        # Linear decrease in speed from v_max to 0
        return max(0, v_f * (1 - density / k_jam))


def cal_travel_time(link_length, density, v_max, k_critical, k_jam):
    """
    Calculate the travel time of a link based on density and link length.

    Args:
        link_length: Length of the link (unit length).
        density: Density of the link (vehicles/unit length).
        v_max: Free flow speed (unit length/unit time).
        k_critical: Critical density (vehicles/unit length).
        k_jam: Jam density (vehicles/unit length).

    Returns:
        Travel time of the link (unit time). Returns infinity if speed is 0 (congestion).
    """
    speed = cal_travel_speed(density, v_max, k_critical, k_jam)
    if speed == 0:
      return float('inf')
    return link_length / speed



# the link model
# class Link:
#     def __init__(self, link_id, start_node, end_node,
#                  length, width, capacity, free_flow_speed,
#                  k_critical, k_jam,  unit_time=10, simulation_steps=100):
#         # Static attributes
#         self.link_id = link_id
#         self.start_node = start_node
#         self.end_node = end_node
#         self.length = length
#         self.width = width
#         self.area = length * width
#         self.capacity = capacity
#         self.free_flow_speed = free_flow_speed
#         self.current_speed = free_flow_speed
#         self.travel_time = length / free_flow_speed  # Tau
#         self.unit_time = unit_time
#         self.free_flow_tau = round(self.travel_time / unit_time)  # Tau in time steps
#
#         # Dynamic attributes
#         self.inflow = np.zeros(simulation_steps)  # List of U_i(t) over time
#         self.cumulative_inflow = np.zeros(simulation_steps)  # List of U_i(t) over time
#         self.outflow = np.zeros(simulation_steps)  # List of V_i(t) over time (Q(t))
#         self.cumulative_outflow = np.zeros(simulation_steps)
#         self.num_pedestrians = np.zeros(simulation_steps)
#         self.density = np.zeros(simulation_steps)
#         self.speed = np.zeros(simulation_steps)
#         self.outflow_buffer = dict()
#         self.gamma = 2e-2
#
#         self.k_critical = k_critical
#         self.k_jam = k_jam
#
#     def cal_receiving_flow(self, time_step: int):
#         pass
#
#     def get_current_outflow(self, time_step):
#         # naive approach
#         if time_step in self.outflow_buffer.keys():
#             q_t = self.outflow_buffer.pop(time_step, 0)
#         else:
#             q_t = 0
#
#         return q_t
#
#     def update_cum_outflow(self, time_step):
#         # naive approach
#         q_t = self.get_current_outflow(time_step)
#         self.outflow[time_step] = q_t
#         self.cumulative_outflow[time_step] = self.cumulative_outflow[time_step - 1] + q_t
#
#     def update_cum_inflow(self, inflow, time_step):
#         self.inflow[time_step] = inflow
#         self.cumulative_inflow[time_step] = self.cumulative_inflow[time_step - 1] + inflow
#
#     def update_speeds(self, time_step):
#         """
#         Update the speed of the link based on the density
#         :param time_step:
#         :return:
#         """
#         num_peds = self.inflow[time_step] - self.outflow[time_step]
#         self.num_pedestrians[time_step] = self.num_pedestrians[time_step - 1] + num_peds
#         self.density[time_step] = self.num_pedestrians[time_step] / self.length
#
#         # speed = cal_travel_speed(self.density[time_step], self.free_flow_speed, k_critical=self.k_critical, k_jam=self.k_jam)
#         speed = self.free_flow_speed
#         if speed == 0:
#           self.travel_time =  float('inf')
#         else:
#             self.travel_time = self.length / speed
#         self.speed[time_step] = speed
#
#
#     def update_outflow_buffer(self, time_step, method='naive'):
#         """
#         Calculate the arrival time t' that current sending flow will arrive at the end of the link. and store the outflow in the buffer
#         :param method:
#         :param time_step:
#         :return:
#         """
#         if method == 'naive':
#             if self.travel_time != float('inf'):
#                 tau = (round(self.travel_time / self.unit_time))
#                 t_prime = time_step + tau
#                 # naive approach calculate the sending flow
#                 sending_flow = self.inflow[time_step]
#                 # sending_flow = self.cumulative_inflow[time_step] - self.cumulative_outflow[time_step]
#                 if t_prime not in self.outflow_buffer.keys():
#                     self.outflow_buffer[t_prime] = 0
#                 self.outflow_buffer[t_prime] += sending_flow
#
#         elif method == 'diffusion':
#             # diffusion approach
#             F = 1/ (1 + self.gamma * self.travel_time)
#             sending_flow = (F * self.inflow[time_step] + F * (1 - F) * self.inflow[time_step - 1] +
#                             F * (1 - F) ** 2 * self.inflow[time_step - 2] +
#                             F * (1 - F) ** 3 * self.inflow[time_step - 3])
#             self.outflow_buffer[time_step + self.free_flow_tau] = sending_flow
#             # tau = (round(self.travel_time / self.unit_time))
#             # t_prime = time_step + tau
#             # if t_prime not in self.outflow_buffer.keys():
#             #     self.outflow_buffer[t_prime] = 0
#             # self.outflow_buffer[t_prime] += sending_flow


class Link:
    def __init__(self, link_id, start_node, end_node,
                 length, width, free_flow_speed,
                 k_critical, k_jam,  unit_time=10, simulation_steps=100):
        # Static attributes
        self.link_id = link_id # string
        self.start_node = start_node
        self.end_node = end_node
        self.length = length
        self.width = width
        self.area = length * width
        self.free_flow_speed = free_flow_speed
        self.capacity = self.free_flow_speed * k_critical
        self.k_jam = k_jam
        self.k_critical = k_critical
        # self.shockwave_speed = self.free_flow_speed - self.capacity / self.k_jam
        self.shockwave_speed = self.capacity / (self.k_jam - self.k_critical)
        self.current_speed = free_flow_speed
        self.travel_time = length / free_flow_speed  # Tau
        self.unit_time = unit_time
        self.free_flow_tau = round(self.travel_time / unit_time)  # Tau in time steps

        # Dynamic attributes
        self.inflow = np.zeros(simulation_steps)  # List of U_i(t) over time
        self.cumulative_inflow = np.zeros(simulation_steps)  # List of U_i(t) over time
        self.outflow = np.zeros(simulation_steps)  # List of V_i(t) over time (Q(t))
        self.cumulative_outflow = np.zeros(simulation_steps)
        self.num_pedestrians = np.zeros(simulation_steps)
        self.density = np.zeros(simulation_steps)
        self.speed = np.zeros(simulation_steps)
        self.outflow_buffer = dict()
        self.gamma = 2e-2 # diffusion parameter
        self.receiving_flow = []

    def update_speeds(self, time_step):
        """
        Update the speed of the link based on the density
        :param time_step:
        :return:
        """
        num_peds = self.inflow[time_step] - self.outflow[time_step]
        if num_peds < 0:
            print(num_peds, self.inflow[time_step], self.outflow[time_step], time_step)
        self.num_pedestrians[time_step] = self.num_pedestrians[time_step - 1] + num_peds
        self.density[time_step] = self.num_pedestrians[time_step] / self.length

        speed = cal_travel_speed(self.density[time_step], self.free_flow_speed, k_critical=self.k_critical, k_jam=self.k_jam)
        # speed = self.free_flow_speed
        if speed == 0:
          self.travel_time =  float('inf')
        else:
            self.travel_time = self.length / speed
        self.speed[time_step] = speed

    def cal_sending_flow(self, time_step: int):
        if self.travel_time == float('inf'):
            return 0

        tau = round(self.travel_time / self.unit_time)
        if time_step - tau < 0:
            sending_flow = 0
        else:
            sending_flow_boundary = self.cumulative_inflow[time_step - tau] - self.cumulative_outflow[time_step - 1]
            sending_flow_max = self.k_critical * self.free_flow_speed * self.unit_time
            sending_flow = min(sending_flow_boundary, sending_flow_max)

        return sending_flow

    def cal_receiving_flow(self, time_step: int):
        if time_step - round(self.length/(self.shockwave_speed * self.unit_time)) < 0:
            receiving_flow_boundary = self.k_jam * self.length
        else:
            receiving_flow_boundary = (self.cumulative_outflow[time_step - round(self.length/(self.shockwave_speed * self.unit_time))]
                              + self.k_jam * self.length - self.cumulative_inflow[time_step - 1])

        receiving_flow_max = self.k_critical * self.free_flow_speed * self.unit_time
        receiving_flow = min(receiving_flow_boundary, receiving_flow_max)
        # if receiving_flow < 30:
        #     print(receiving_flow, receiving_flow_boundary, receiving_flow_max, time_step)

        self.receiving_flow.append(receiving_flow)
        return receiving_flow

    def update_cum_outflow(self, q_ij, time_step):
        self.outflow[time_step] = q_ij
        self.cumulative_outflow[time_step] = self.cumulative_outflow[time_step - 1] + q_ij

    def update_cum_inflow(self, q_ij, time_step):
        self.inflow[time_step] = q_ij
        self.cumulative_inflow[time_step] = self.cumulative_inflow[time_step - 1] + q_ij

class Node:
    def __init__(self, node_id):
        self.node_id = node_id
        self.incoming_links = []
        self.outgoing_links = []
        self.turning_fractions = None
        self.q_ij = None
        self.w = 1e-2
        self.source_num = None
        self.dest_num = None
        self.edge_num = None
        self.demand = None
        self.A_ub = None
        self.A_eq = None

    def init_node(self):
        """Initializes node-specific attributes based on the type."""
        self.source_num = len(self.incoming_links)
        self.dest_num = len(self.outgoing_links)
        if isinstance(self, RegularNode) or isinstance(self, OneToOneNode):
            self.edge_num = self.dest_num * self.source_num

        elif isinstance(self, OriginNode):
            # self.source_num = 1
            self.dest_num = len(self.outgoing_links)
            self.edge_num = 1 * self.dest_num

        elif isinstance(self, DestinationNode):
            self.source_num = len(self.incoming_links)
            # self.dest_num = 1
            self.edge_num = self.source_num * 1

    def update_turning_fractions(self, turning_fractions: list):
        self.turning_fractions = turning_fractions

    def get_matrix_A(self):
        if self.edge_num > 1:
            # row_num = self.source_num + self.dest_num
            # self.A_ub = np.zeros((row_num, self.edge_num + 2 * self.edge_num))
            #
            # for i in range(self.source_num):
            #     self.A_ub[i, i:i + self.dest_num] = 1
            #
            # for j in range(self.dest_num):
            #     for k in range(self.source_num):
            #         self.A_ub[self.source_num + j, j + k * self.dest_num] = 1
            #
            # # turning fractions constraints
            # self.update_matrix_A_eq(self.turning_fractions)
            row_num = self.source_num + self.dest_num
            self.A_ub = np.zeros((row_num, self.edge_num + 2 * self.edge_num))

            for i in range(self.source_num):
                self.A_ub[i, i * self.dest_num : (i+1) * self.dest_num] = 1

            for j in range(self.dest_num):
                for k in range(self.source_num):
                    self.A_ub[self.source_num + j, j + k * self.dest_num] = 1

            # turning fractions constraints
            # if self.A_eq is not None:
            self.update_matrix_A_eq(self.turning_fractions)

    def update_matrix_A_eq(self, turning_fractions: list):
        """
        Update hte turning fractions of A_eq
        :return:
        """
        self.turning_fractions = turning_fractions
        if self.dest_num > 1:
            self.A_eq = np.zeros((self.edge_num, self.edge_num + 2 * self.edge_num))
            for l in range(self.edge_num):
                if l % self.dest_num == 0:
                    start_ind = l
                self.A_eq[l, start_ind : start_ind + self.dest_num] = turning_fractions[l] * np.ones(self.dest_num)
                self.A_eq[l, l] = turning_fractions[l] - 1
                self.A_eq[l, self.edge_num + l * 2 : self.edge_num + (l+1) * 2] = np.array([1, -1])  # the penalty term

    def update_links(self, time_step):
        """Update the upstream link's downstream cumulative outflow
        q_ij -->> [S1, S2, R1, R2, R3], already sum up the flows from/to the same link
        """
        assert self.q_ij is not None

        for l, link in enumerate(self.incoming_links):
            # inflow = np.sum(self.q_ij[l * self.dest_num : (l + 1) * self.dest_num])
            inflow = self.q_ij[l]
            link.update_cum_outflow(inflow, time_step)
            link.update_speeds(time_step)

        for m, link in enumerate(self.outgoing_links):
            # outflow = np.sum(self.q_ij[m + self.source_num])
            outflow = self.q_ij[self.source_num + m]
            link.update_cum_inflow(outflow, time_step)
            link.update_speeds(time_step)

class OriginNode(Node):
    def __init__(self, node_id):
        super().__init__(node_id)

    def assign_flows(self, time_step):
        assert self.demand is not None
        s = np.array([self.demand[time_step]]) # for origin node, the demand is the outflow of its upstream link already.
        r = np.zeros(self.dest_num)
        for j, l in enumerate(self.outgoing_links):
            r[j] = l.cal_receiving_flow(time_step)

        if self.edge_num == 1:
            self.q_ij = np.array([np.min([s[0], r[0]])]) # this case s and r are scalars
            self.update_links(time_step)
            return

        # solve the linear programming problem
        w = self.w * np.ones(2 * self.edge_num)
        c = -1 * np.ones(self.edge_num)
        c = np.concatenate((c, w))
        b_ub = np.concatenate((s, r))
        res = linprog(c, A_ub=self.A_ub, A_eq=self.A_eq, b_ub=b_ub, b_eq=np.zeros(self.edge_num))

        if res.success:
            # flows = res.x[:self.edge_num]
            self.q_ij = np.floor(self.A_ub @ res.x)
            # total_flow = -result.fun
            self.update_links(time_step)
        return

class DestinationNode(Node):
    def __init__(self, node_id):
        super().__init__(node_id)

    def assign_flows(self, time_step):
        s = np.zeros(self.source_num)
        for i, l in enumerate(self.incoming_links):
            s[i] = l.cal_sending_flow(time_step)
        self.q_ij = s
        self.update_links(time_step)
        return

class OneToOneNode(Node):
    def __init__(self, node_id):
        super().__init__(node_id)

    def assign_flows(self, time_step):
        s = self.incoming_links[0].cal_sending_flow(time_step)
        r = self.outgoing_links[0].cal_receiving_flow(time_step)
        self.q_ij = np.array(min(s, r))
        self.update_links(time_step)
        return

class RegularNode(Node):
    def __init__(self, node_id):
        super().__init__(node_id)

    # def get_matrix_A(self):
    #     row_num = self.source_num + self.dest_num
    #     # self.A_ub = np.zeros((row_num, self.edge_num + 2 * self.edge_num))
    #     self.A_ub = np.zeros((row_num, self.edge_num + 2 * self.edge_num))
    #
    #     for i in range(self.source_num):
    #         self.A_ub[i, i * self.dest_num : (i+1) * self.dest_num] = 1
    #
    #     for j in range(self.dest_num):
    #         for k in range(self.source_num):
    #             self.A_ub[self.source_num + j, j + k * self.dest_num] = 1
    #
    #     # turning fractions constraints
    #     self.update_matrix_A_eq(self.turning_fractions)


    def assign_flows(self, time_step):
        s = np.zeros(self.source_num)
        r = np.zeros(self.dest_num)
        # cal sending flow of upstream links
        for i, l in enumerate(self.incoming_links):
            s[i] = l.cal_sending_flow(time_step)
        # cal receiving flow of downstream links
        for j, l in enumerate(self.outgoing_links):
            r[j] = l.cal_receiving_flow(time_step)

        # solve the linear programming problem
        w = self.w * np.ones(2 * self.edge_num)
        c = -1 * np.ones(self.edge_num)
        c = np.concatenate((c, w))
        b_ub = np.concatenate((s, r))
        if self.A_eq is not None:
            res = linprog(c, A_ub=self.A_ub, A_eq=self.A_eq, b_ub=b_ub, b_eq=np.zeros(self.edge_num))
        else:
            res = linprog(c, A_ub=self.A_ub, b_ub=b_ub)

        if res.success:
            # flows = res.x[:self.edge_num]
            # self.q_ij = flows
            self.q_ij = np.floor(self.A_ub @ res.x)
            # total_flow = -result.fun
            self.update_links(time_step)
        return


class Network:
    def __init__(self, adjacency_matrix: np.array, link_params: dict):
        """
        Initialize the network, with nodes and links according to the adjacency matrix
        """
        self.adjacency_matrix = adjacency_matrix
        self.nodes = []
        self.links = {}          #use dict to store link, key is tuple(start_node_id, end_node_id)
        self.link_params = link_params
        self.simulation_steps = link_params['simulation_steps']
        self.init_nodes_and_links()

    # def update_turning_fractions(self, new_turning_fractions):
    #     for node in self.nodes:
    #         if node.source_num > 0 and node.dest_num > 0:
    #             if node.turning_fractions is None:
    #                 phi = 1/node.source_num
    #                 node.turning_fractions = np.ones(node.edge_num) * phi
    #             else:
    #                 node.turning_fractions = new_turning_fractions

    def init_nodes_and_links(self, lam=10):
        """
        Initialize nodes and links based on the adjacency matrix.
        """
        num_nodes = self.adjacency_matrix.shape[0]

        # Create nodes
        for i in range(num_nodes):
            self.nodes.append(Node(node_id=i))

        # Create links
        for i in range(num_nodes):
            for j in range(num_nodes):
                if self.adjacency_matrix[i, j] == 1:
                    link_id = f"{i}_{j}"
                    link = Link(link_id, self.nodes[i], self.nodes[j], **self.link_params)
                    self.links[(i, j)] = link
                    self.nodes[i].outgoing_links.append(link)
                    self.nodes[j].incoming_links.append(link)

        for node in self.nodes:
            if not node.incoming_links:
                # Origin node
                node.__class__ = OriginNode  # Change the class dynamically
                node.demand = np.random.poisson(lam=30, size=self.simulation_steps)
            elif not node.outgoing_links:
                # Destination node
                node.__class__ = DestinationNode
            elif len(node.outgoing_links) == 1 and len(node.incoming_links) == 1:
                node.__class__ = OneToOneNode
            else:
                # Regular node
                node.__class__ = RegularNode
            node.init_node() #Initialize the node after determine the type

    def update_transition_probability(self, new_turning_fractions_matrix):
        pass

    def network_loading(self, time_step):
        for node in self.nodes:
            if node.turning_fractions is None:
                phi = 1/node.dest_num if node.dest_num > 1 else 1
                turning_fractions = np.ones(node.edge_num) * phi
                node.update_turning_fractions(turning_fractions)
            else:
                # node.update_turning_fractions(new_turning_fractions)
                pass

            if isinstance(node, OriginNode) or isinstance(node, RegularNode):
                node.get_matrix_A()  # only for RegularNode and OriginNode with multiple outgoing links
                node.assign_flows(time_step)
            else:
                node.assign_flows(time_step)

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
        node_options = {"node_size": 700, "node_color": "skyblue", "font_size": 15, "font_weight": "bold"}
        edge_options = {"arrowstyle": "-|>", "arrowsize": 20}  # Add arrows for directed graph

        # Draw nodes and edges
        nx.draw(graph, pos, with_labels=True, **node_options)
        nx.draw_networkx_edges(graph, pos, **edge_options) # Draw edges with arrows

        # Draw edge labels
        edge_labels = nx.get_edge_attributes(graph, 'label')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10)

        plt.tight_layout() # Prevents labels from being cut off
        plt.show()

if __name__ == "__main__":
    # generate a list of demand with a poisson distribution
    np.random.seed(0)
    # demand = np.random.poisson(lam=35, size=50)
    # demand2 = np.random.poisson(lam=30, size=50)
    demand = np.random.poisson(lam=25, size=50)
    demand2 = np.random.poisson(lam=30, size=50)
    demand3 = np.random.poisson(lam=10, size=60)
    demand = np.concatenate((demand, demand2, demand3))
    # Define the network
    adj = np.array([[0, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 0, 1, 1],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]])

    params = {
        'length': 50,
        'width': 1,
        'free_flow_speed': 1.5,
        'k_critical': 2,
        'k_jam': 10,
        'unit_time': 10,
        'simulation_steps': 200,
        # 'lam': 25
    }

    # Initialize the network
    network = Network(adj, params)
    network.visualize()

    # link1 = Link(1, node1, node2, 60, 1, 2*1.5, 1.5, 2, k_jam=10, unit_time=10, simulation_steps=len(demand)) # capacity = free_flow_speed * k_critical


    # Run the simulation
    for t in range(1, params['simulation_steps']):
        network.network_loading(t)
        # link1.update_speeds(t)


# draw inflow list and outflow list of link1
import matplotlib.pyplot as plt
link_id = (1, 2)
plt.plot(network.links[link_id].inflow, label='inflow')
plt.plot(network.links[link_id].outflow, label='outflow')
# plt.plot(network.links[link_id].outflow, label='outflow')
# plt.plot(network.links[link_id].receiving_flow, label='receiving_flow')
# plt.plot(network.links[link_id].num_pedestrians, label='num_pedestrians')
plt.legend()
plt.show()

# plot density and speed
plt.plot(network.links[link_id].density, label='density')
plt.plot(network.links[link_id].speed, label='speed')
plt.legend()
plt.show()

# plot cumulative inflow and outflow
plt.plot(network.links[link_id].cumulative_inflow, label='cumulative_inflow')
plt.plot(network.links[link_id].cumulative_outflow, label='cumulative_outflow')
plt.legend()
plt.show()
