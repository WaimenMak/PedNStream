import numpy as np
# from .link import BaseLink
from .solver import NodeFlowSolver
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class NodeConfig:
    """Static configuration for a Node. Only holds values known before simulation starts.

    Dynamic/runtime state (q, A_ub, mask, source_num, dest_num, edge_num,
    virtual links, ods_in_turns) lives on Node itself.
    """
    id: int
    # node_type: str = "regular"          # "regular" or "onetoone"
    gate_width: Optional[float] = None
    turning_fractions: Optional[np.ndarray] = None
    demand: Optional[np.ndarray] = None  # demand profile for origin node
    M: Optional[float] = 1e6            # penalty term for destination node
    w: Optional[float] = 1e-2           # penalty term for turning fractions
    is_controller: bool = False         # whether this node is a controller, can be used to populate Network's controllers list


class Node:
    """Represents a node in the traffic network."""

    def __init__(self, node_config: NodeConfig):
        """Initialize the Node with its static configuration."""
        # --- Static config (from NodeConfig) ---
        self.id = node_config.id
        self.turning_fractions = node_config.turning_fractions  # 1D array, length = edge_num
        self.demand = node_config.demand              # demand profile for origin node
        self.M = node_config.M                        # penalty for destination node
        self.w = node_config.w                        # penalty for turning fractions
        self.gate_width = node_config.gate_width

        # --- Link containers (populated externally by Network) ---
        self.incoming_links: list = []
        self.outgoing_links: list = []
        self.virtual_incoming_link = None
        self.virtual_outgoing_link = None

        # --- Runtime state (computed during simulation) ---
        self.q = None           # flows for each edge, set in assign_flows
        self.A_ub = None        # LP constraint matrix, built in get_matrix_A
        self.mask = None        # boolean mask, built in init_mask
        self.ods_in_turns = {}  # populated on-the-fly during path-finding/flow assignment
        self._type = ""     # "onetoone" or "regular"

        self.is_controller = node_config.is_controller # whether this node is a controller, can be used to populate Network's controllers list
    # ------------------------------------------------------------------
    # Properties: always in sync with the actual link lists — no stale
    # state and no need to call init_node() just to read these values.
    # ------------------------------------------------------------------

    @property
    def type(self):
        """Return the type of the node."""
        return self._type
    
    @type.setter
    def type(self, value) -> None:
        """Set the type of the node based on the adjacency matrix and origin/destination nodes."""
        try:
            adjacency_matrix, origin_nodes, destination_nodes = value 
        except ValueError:
            raise ValueError("Value must be a tuple of (adjacency_matrix, origin_nodes, destination_nodes)")
        else:
            incoming_links = np.sum(adjacency_matrix[: self.id])
            outgoing_links = np.sum(adjacency_matrix[self.id :])
        
            # node type rules
            if incoming_links >= 2 and outgoing_links >= 2:
                if self.id in origin_nodes or self.id in destination_nodes:
                    self._type = "regular"
                    # Create virtual Node
                    # FIXME: move creation of virtual links to be created in a separated method.
                    # self._create_origin_destination(node_config) # These Are Links. 
                else:  # do not create virtual links 
                    self._type = "onetoone"  

            elif incoming_links == 1 and outgoing_links ==1:
                self._type = "onetoone"
                # create virtual Node
                # self._create_origin_destination(node_config)
            # CONTINUE HERE: test this works as expected.
            return None

    @property
    def source_num(self) -> int:
        """Number of incoming links (sources at this node)."""
        return len(self.incoming_links)

    @property
    def dest_num(self) -> int:
        """Number of outgoing links (destinations at this node)."""
        return len(self.outgoing_links)

    @property
    def edge_num(self) -> int:
        """Number of turning edges (excludes U-turn from each source)."""
        return self.dest_num * self.source_num - self.source_num

    def _create_virtual_link(self, node_id, direction, is_incoming, params: dict):
        """Helper method to create virtual links for origin and destination nodes"""
        link = BaseLink(
            link_id=f"virtual_{direction}_{node_id}",
            start_node=self if not is_incoming else None, # TODO move logic to Network creation.
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

    def init_mask(self):
        """Build the boolean mask for optimal flow assignment once all links are attached.

        Must be called after incoming/outgoing links are fully populated.
        source_num / dest_num / edge_num are live properties, so this is
        the only one-time computation that still needs an explicit trigger.
        """
        self.mask = np.ones([self.source_num, self.source_num], dtype=bool)
        np.fill_diagonal(self.mask, False)

    def get_matrix_A(self):
        """
        Get the matrix A_ub for the linear programming problem, only computed once.
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
                hasattr(self, "virtual_incoming_link") # TODO: this can be substituted by a type check for VituralLink
                and l == self.virtual_incoming_link
            ):
                s[i] = self.demand[time_step - 1]
            else:
                s[i] = l.cal_sending_flow(time_step - 1)

        # Calculate receiving flows
        for j, l in enumerate(self.outgoing_links):
            if (
                hasattr(self, "virtual_outgoing_link") # TODO: this can be substituted by a type check for VituralLink
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
