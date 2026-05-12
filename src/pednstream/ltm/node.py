import numpy as np
from .link import BaseLink
from .solver import NodeFlowSolver


class Node:
    def __init__(self, node_id, node_type: str = "regular"):
        self.node_id = node_id
        self.node_type = node_type  # "onetoone" or "regular"
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

    def update_links(self, time_step):
        """Update the upstream link's downstream cumulative outflow
        q -->> [S1, S2, R1, R2, R3], already sum up the flows from/to the same link
        """
        assert self.q is not None
        # whether length of q is number of edges
        assert len(self.q) == self.source_num + self.dest_num

        for l, link in enumerate(self.incoming_links):
            inflow = self.q[
                l
            ]  # here inflow is from the perspective of the node, which is the outflow of the link
            link.update_cum_outflow(inflow, time_step)

        for m, link in enumerate(self.outgoing_links):
            outflow = self.q[self.source_num + m]
            link.update_cum_inflow(outflow, time_step)

    def assign_flows(self, time_step: int, type="classic"):
        """
        Get the sending and receiving flows constraints. time_step starts from 1.
        """
        s = np.zeros(self.source_num)
        r = np.zeros(self.dest_num)

        # Calculate sending flows
        for i, l in enumerate(self.incoming_links):
            if (
                hasattr(self, "virtual_incoming_link")
                and l == self.virtual_incoming_link
            ):
                s[i] = self.demand[time_step - 1]
            else:
                s[i] = l.cal_sending_flow(time_step - 1)

        # Calculate receiving flows
        for j, l in enumerate(self.outgoing_links):
            if (
                hasattr(self, "virtual_outgoing_link")
                and l == self.virtual_outgoing_link
            ):
                r[j] = self.M
            else:
                reverse_sending_flow = l.reverse_link.sending_flow[time_step - 1].copy()
                # raise Warning if reverse_sending_flow is negative
                if reverse_sending_flow < 0:
                    print(reverse_sending_flow)
                    raise Warning(
                        f"Negative reverse sending flow detected at time step {time_step}: {reverse_sending_flow}"
                    )

                r[j] = l.cal_receiving_flow_with_reverse(
                    time_step - 1, reverse_sending_flow
                )
                l.receiving_flow[time_step - 1] = r[j]

        # raise Warining if s and r has negative values
        if np.any(s < 0) or np.any(r < 0):
            raise Warning(
                f"Negative flows detected at time step {time_step}: s={s}, r={r}"
            )

        # Delegate flow computation to the standalone solver
        q = NodeFlowSolver.solve(self, s, r, type=type)
        if q is not None:
            self.q = q
        self.update_links(time_step)
