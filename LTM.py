# -*- coding: utf-8 -*-
# @Time    : 26/11/2024 12:10
# @Author  : mmai
# @FileName: draft
# @Software: PyCharm

from scipy.optimize import linprog
import numpy as np

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
                 length, width, capacity, free_flow_speed,
                 k_critical, k_jam,  unit_time=10, simulation_steps=100):
        # Static attributes
        self.link_id = link_id # string
        self.start_node = start_node
        self.end_node = end_node
        self.length = length
        self.width = width
        self.area = length * width
        self.capacity = capacity
        self.free_flow_speed = free_flow_speed
        self.shockwave_speed = capacity / (k_jam - k_critical)
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

    def cal_sending_flow(self, time_step: int):
        sending_flow = self.inflow[time_step]
        return sending_flow

    def cal_receiving_flow(self, time_step: int):
        receiving_flow = self.outflow[time_step]
        return receiving_flow

    def update_cum_outflow(self, q_ij, time_step):
        self.outflow[time_step] = q_ij
        self.cumulative_outflow[time_step] = self.cumulative_outflow[time_step - 1] + q_ij

    def update_cum_inflow(self, q_ij, time_step):
        self.inflow[time_step] = q_ij
        self.cumulative_inflow[time_step] = self.cumulative_inflow[time_step - 1] + q_ij



class Node:
    def __init__(self, node_id: int):
        self.node_id = node_id # string

        self.turning_fractions = None
        self.q_ij = np.zeros(self.edge_num)  # flow from source to dest at time t
        self.A_eq = None
        self.A_ub = None
        self.w = 1e-2  # weight of the turning fractions
        self.incoming_links = []
        self.outgoing_links = []

    def init_node(self):
        self.dest_num = len(self.incoming_links)
        self.source_num = len(self.outgoing_links)
        self.edge_num = self.dest_num * self.source_num

    def define_type(self):
        """
        Source, dest, or Intermediate, Source: only outgoing links, dest: only incoming links, Intermediate: both,
        Source node provide demand, dest node assume the receiving flow of the upstream link is infinite
        :return:
        """
        pass

    def update_turning_fractions(self, turning_fractions: list):
        pass

    def get_matrix_A(self):
        row_num = self.edge_num + self.source_num + self.dest_num
        self.A_ub = np.zeros((row_num, self.edge_num + 2 * self.edge_num))
        self.A_eq = np.zeros((self.edge_num, self.edge_num + 2 * self.edge_num))

        for i in range(self.source_num):
            self.A_ub[i, i:i + self.dest_num] = 1

        for j in range(self.dest_num):
            for k in range(self.dest_num):
                self.A_ub[self.source_num + j, j + k * self.dest_num] = 1

        # turning fractions constraints
        self.update_matrix_A_eq(self.turning_fractions)

    def update_matrix_A_eq(self, turning_fractions: list):
        """
        Update hte turning fractions of A_eq
        :return:
        """
        for l in range(self.edge_num):
            self.A_eq[self.source_num + self.dest_num + l, l + l * self.dest_num] = turning_fractions[l] * np.ones(self.dest_num)
            self.A_eq[self.source_num + self.dest_num + l, l] = turning_fractions[l] - 1
            self.A_eq[self.source_num + self.dest_num + l, self.edge_num + l * self.dest_num : self.edge_num + (l + 1) * self.dest_num] = np.array([1, -1])


    def assign_flows(self, time_step):
        s = np.zeros(self.source_num)
        r = np.zeros(self.dest_num)
        # cal sending flow of upstream links
        for i, l in enumerate(self.incoming_links):
            s[i] = l.cal_sending_flow(time_step)
        # cal receiving flow of downstream links
        for j, l in enumerate(self.outgoing_links):
            r[i] = l.cal_receiving_flow(time_step)

        # solve the linear programming problem
        w = self.w * np.ones(2 * self.edge_num)
        c = -1 * np.ones(self.edge_num)
        c = np.concatenate((c, w))
        b_ub = np.concatenate((s, r))
        res = linprog(c, A_ub=self.A_ub, A_eq=self.A_eq, b_ub=b_ub, b_eq=np.zeros(self.edge_num))

        if res.success:
            flows = res.x
            self.q_ij = flows
            # total_flow = -result.fun
            self.update_links(time_step)


    def update_links(self, time_step):
        """
        Update the upstream link's downstream cumulative outflow
        :return:
        """
        for l, link in enumerate(self.incoming_links):
            link.update_cum_outflow(self.q_ij[l], time_step)

        for l, link in enumerate(self.outgoing_links):
            link.update_cum_inflow(self.q_ij[l + self.source_num], time_step)


class Network:
    def __init__(self, adjacency_matrix: np.array, link_params: dict):
        """
        Initialize the network, with nodes and links according to the adjacency matrix
        """
        self.adjacency_matrix = adjacency_matrix
        self.nodes = []
        self.links = {}          #use dict to store link, key is tuple(start_node_id, end_node_id)
        self.link_params = link_params
        self.init_nodes_and_links()

    def update_turning_fractions(self, new_turning_fractions):
        for node in self.nodes:
            if node.source_num > 0 and node.dest_num > 0:
                if node.turning_fractions is None:
                    phi = 1/node.source_num
                    node.turning_fractions = np.ones(node.edge_num) * phi
                else:
                    node.turning_fractions = new_turning_fractions

    def init_nodes_and_links(self):
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

        for i in range(num_nodes):
            self.nodes[i].init_node()

    def update_transition_probability(self, new_turning_fractions_matrix):
        pass

    def network_loading(self):
        pass

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
    adj = np.array([[0, 0, 1, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1],
                     [0, 0, 0, 0]])

    params = {
        'length': 60,
        'width': 1,
        'capacity': 3,
        'free_flow_speed': 1.5,
        'k_critical': 2,
        'k_jam': 10,
        'unit_time': 10,
        'simulation_steps': 100
    }

    # Initialize the network
    network = Network(adj, params)



    node1 = Node(1)
    node2 = Node(2)
    link1 = Link(1, node1, node2, 60, 1, 2*1.5, 1.5, 2, k_jam=10, unit_time=10, simulation_steps=len(demand)) # capacity = free_flow_speed * k_critical
    # link2 = Link(2, node2, node1, 60, 1, 500, 1.5, 10)
    node1.outgoing_links.append(link1)
    node2.incoming_links.append(link1)

    # Run the simulation
    for t in range(1, 150):
        # Update the demand
        # node1.update_demand(100)
        # node2.update_demand(100)

        # Update the inflow

        link1.update_cum_inflow(demand[t], t)
        link1.update_cum_outflow(t)
        link1.update_speeds(t)
        # link1.update_outflow_buffer(t, method='diffusion')
        link1.update_outflow_buffer(t, method='naive')



# draw inflow list and outflow list of link1
import matplotlib.pyplot as plt
plt.plot(link1.inflow, label='inflow')
plt.plot(link1.outflow[4:], label='outflow')
plt.plot(link1.num_pedestrians, label='num_pedestrians')
plt.legend()
plt.show()

# plot density and speed
plt.plot(link1.density, label='density')
plt.plot(link1.speed, label='speed')
plt.legend()
plt.show()

# plot cumulative inflow and outflow
plt.plot(link1.cumulative_inflow, label='cumulative_inflow')
plt.plot(link1.cumulative_outflow, label='cumulative_outflow')
plt.legend()
plt.show()
