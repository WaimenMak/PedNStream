import numpy as np
from scipy.optimize import linprog
from .link import BaseLink
from abc import ABC, abstractmethod


class Node(ABC):
    def __init__(self, node_id):
        self.node_id = node_id
        self.incoming_links = []
        self.outgoing_links = []
        self.turning_fractions = None  # 1D array, the length is the number of edges
        self.mask = None
        self.q = None
        self.w = 1e-2  # penalty term for turning fractions
        self.source_num = None
        self.dest_num = None
        self.edge_num = None
        self.A_ub = None
        self.A_eq = None
        self.virtual_incoming_link = None
        self.virtual_outgoing_link = None
        self.M = 1e6  # for destination node, large constant for receiving flow
        self.demand = None  # for origin node
        self.mask = None  # for regular node, classic update method
        self.ods_in_turns = {}  # for recording the turns in which od pairs

    def _create_virtual_link(self, node_id, direction, is_incoming, params: dict):
        """Helper method to create virtual links for origin and destination nodes"""
        link = BaseLink(
            link_id=f"virtual_{direction}_{node_id}",
            start_node=self if not is_incoming else None,
            end_node=self if is_incoming else None,
            simulation_steps=params["simulation_steps"],
        )
        if is_incoming:
            self.incoming_links.append(link)
            self.virtual_incoming_link = link
        else:
            self.outgoing_links.append(link)
            self.virtual_outgoing_link = link
        return link

    def init_node(self):
        """Initializes node-specific attributes based on the type."""
        # source number is the number of incoming links
        self.source_num = len(self.incoming_links)
        self.dest_num = len(self.outgoing_links)
        self.edge_num = self.dest_num * self.source_num - self.source_num
        self.mask = np.ones([self.source_num, self.source_num], dtype=bool)
        np.fill_diagonal(self.mask, False)


    @abstractmethod
    def solve(self, s, r, type="classic"):
        return


class NodeSolver:
    """
    Interface class that mediates between Node and Link classes.
    Handles flow gathering from links, delegates solving to Node, and updates links with results.
    One NodeSolver instance per Node.
    """

    def __init__(self, node: Node, solve_type: str = "classic"):
        self.node = node
        self.solve_type = solve_type

    @property
    def incoming_links(self):
        return self.node.incoming_links

    @property
    def outgoing_links(self):
        return self.node.outgoing_links

    def gather_flows(self, time_step: int):
        """
        Gather sending and receiving flows from connected links.
        
        Args:
            time_step: Current simulation time step (starts from 1)
            
        Returns:
            tuple: (s, r) where s is sending flows array, r is receiving flows array
        """
        s = np.zeros(self.node.source_num)
        r = np.zeros(self.node.dest_num)

        # Calculate sending flows from incoming links
        for i, link in enumerate(self.incoming_links):
            if (
                hasattr(self.node, "virtual_incoming_link")
                and link == self.node.virtual_incoming_link
            ):
                s[i] = self.node.demand[time_step - 1]
            else:
                s[i] = link.cal_sending_flow(time_step - 1)

        # Calculate receiving flows from outgoing links
        for j, link in enumerate(self.outgoing_links):
            if (
                hasattr(self.node, "virtual_outgoing_link")
                and link == self.node.virtual_outgoing_link
            ):
                r[j] = self.node.M
            else:
                reverse_sending_flow = link.reverse_link.sending_flow[time_step - 1].copy()
                if reverse_sending_flow < 0:
                    print(reverse_sending_flow)
                    raise Warning(
                        f"Negative reverse sending flow detected at time step {time_step}: {reverse_sending_flow}"
                    )

                r[j] = link.cal_receiving_flow_with_reverse(
                    time_step - 1, reverse_sending_flow
                )
                link.receiving_flow[time_step - 1] = r[j]

        return s, r

    def update_links(self, time_step: int):
        """
        Update connected links with the solved flow values from node.q.
        q format: [S1, S2, ..., R1, R2, ...] - flows summed per link.
        
        Args:
            time_step: Current simulation time step
        """
        q = self.node.q
        assert q is not None
        assert len(q) == self.node.source_num + self.node.dest_num

        for idx, link in enumerate(self.incoming_links):
            # From node's perspective: inflow to node = outflow from link
            link.update_cum_outflow(q[idx], time_step)

        for idx, link in enumerate(self.outgoing_links):
            # From node's perspective: outflow from node = inflow to link
            link.update_cum_inflow(q[self.node.source_num + idx], time_step)

    def assign_flows(self, time_step: int, solve_type: str = None):
        """
        Main method: gather flows, solve node, update links.
        
        Args:
            time_step: Current simulation time step (starts from 1)
            solve_type: Override solve type ("classic" or "optimal"), uses default if None
        """
        if solve_type is None:
            solve_type = self.solve_type

        s, r = self.gather_flows(time_step)

        if np.any(s < 0) or np.any(r < 0):
            raise Warning(
                f"Negative flows detected at time step {time_step}: s={s}, r={r}"
            )

        self.node.solve(s, r, type=solve_type)
        self.update_links(time_step)



class OneToOneNode(Node):
    def __init__(self, node_id):
        super().__init__(node_id)

    def solve(self, s, r, type="classic"):
        """
        q = [S0, S1, R0, R1], S1 and R1 are virtual links
        """
        # Ensure non-negative flows by using maximum of 0 and the minimum flow
        self.q = np.array(
            [
                np.min([s[0], r[1]]),
                np.min([s[1], r[0]]),
                np.min([s[1], r[0]]),
                np.min([s[0], r[1]]),
            ]
        )
        if np.any(self.q < 0):
            raise Warning(f"Negative flows detected: {self.q}")
        return


class RegularNode(Node):
    def __init__(self, node_id):
        super().__init__(node_id)

    def get_matrix_A(self):
        """
        Get the matrix A_ub for the linear programming problem
        """
        row_num = self.source_num + self.dest_num
        # - source_num for the link from the same source-destination pair, now it is included in the source_num
        self.A_ub = np.zeros(
            (row_num, self.edge_num + self.source_num + 2 * self.edge_num)
        )  # 2 * edge_num for the penalty term

        # set the constraints for the source node
        for i in range(self.source_num):  # S[i]
            e = np.ones(self.dest_num)
            e[i] = 0
            self.A_ub[i, i * (self.dest_num) : (i + 1) * (self.dest_num)] = e

        # set the constraints for the destination node
        for j in range(self.dest_num):  # R[j]
            for k in range(self.source_num):
                if k != j:
                    self.A_ub[self.source_num + j, j + k * (self.dest_num)] = (
                        1  # consider the flow on
                    )
                else:
                    self.A_ub[self.source_num + j, j + k * self.dest_num] = 0

        # remove the columns for the flow from the same source-destination pair
        start_idx = [i * self.dest_num + i for i in range(self.source_num)]
        self.A_ub = np.delete(self.A_ub, start_idx, axis=1)

    def update_matrix_A_eq(self, turning_fractions: np.array):
        """
        Update the turning fractions matrix A_eq more efficiently

        Args:
            turning_fractions: Array of turning fraction values
        """
        self.turning_fractions = turning_fractions
        assert len(turning_fractions) == self.edge_num

        # Initialize A_eq matrix with zeros
        self.A_eq = np.zeros((self.edge_num, self.edge_num + 2 * self.edge_num))

        for i in range(self.edge_num):
            source_idx = i // (self.dest_num - 1)
            start_ind = source_idx * (self.dest_num - 1)

            # Set turning fractions for all destinations from this source
            self.A_eq[i, start_ind : start_ind + self.dest_num - 1] = turning_fractions[
                i
            ]
            # the penalty term for the edge from the same source-destination pair should be 0
            self.A_eq[i, i] = turning_fractions[i] - 1  # the lth column is phi - 1
            self.A_eq[i, self.edge_num + i * 2 : self.edge_num + (i + 1) * 2] = (
                np.array([1, -1])
            )  # the penalty term

    def solve(self, s, r, type="classic"):
        if type == "optimal":
            self.update_matrix_A_eq(
                self.turning_fractions
            )  # update the matrix A_eq for the turning fractions
            # solve the linear programming problem
            w = self.w * np.ones(2 * self.edge_num)  # variables for the penalty term
            c = -1 * np.ones(self.edge_num)  # variables for the flow
            c = np.concatenate((c, w))
            assert self.edge_num > 0
            b_ub = np.concatenate((s, r))

            if self.A_eq is not None:
                res = linprog(
                    c,
                    A_ub=self.A_ub,
                    A_eq=self.A_eq,
                    b_ub=b_ub,
                    b_eq=np.zeros(self.edge_num),
                )
            else:
                res = linprog(c, A_ub=self.A_ub, b_ub=b_ub)
            if res.success:
                flows = self.A_ub @ np.floor(res.x)
                # Ensure non-negative flows and round down to nearest integer
                self.q = np.maximum(0, flows)
            return
        if type == "classic":
            # use the update method originally from LTM paper
            s_tiled = np.tile(s, (self.source_num, 1))
            p = self.turning_fractions.reshape(self.dest_num, self.source_num - 1)
            # Add zero diagonal elements to make it a square matrix (m x m)
            p_square = np.zeros((self.dest_num, self.source_num))
            # Fill non-diagonal elements with the reshaped turning fractions
            p_square[self.mask] = p.flatten()
            p = p_square

            weighted_s_frac = p * s_tiled.T  # 2
            row_sums = np.sum(weighted_s_frac, axis=0, keepdims=True)  # 2
            weighted_s = weighted_s_frac / np.where(row_sums != 0, row_sums, 1e-5)  # 2
            weighted_sr = r * weighted_s  # 2

            flows_list = []
            for i in range(self.edge_num):
                source_idx = i // (self.dest_num - 1)
                destination_idx = i % (self.dest_num - 1)
                g_ij = min(
                    self.turning_fractions[i] * s[source_idx],
                    weighted_sr[source_idx][self.mask[source_idx, :]][destination_idx],
                )  # do not use min # 2
                flows_list.append(g_ij)
            flows = self.A_ub[:, : self.edge_num] @ np.floor(np.array(flows_list))
            self.q = np.maximum(0, flows)
            return
        else:
            raise ValueError(f"Invalid type: {type}")
