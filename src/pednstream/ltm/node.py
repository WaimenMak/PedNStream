import numpy as np
from collections import defaultdict
from scipy.optimize import linprog
from .link import BaseLink
from .solver import NodeFlowSolver
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List
from pednstream.ltm import DemandGenerator
from typing import Optional

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
        self._demand: np.ndarray = field(default_factory=lambda: np.zeros(1))  # Default to an array of zeros if not provided
        self.generator: Optional[DemandGenerator] = None

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
        self.node_turn_probs = {}
        self.turns_distances = {}
        self.up_od_probs = defaultdict(lambda: defaultdict(int))
        self.current_turn_probs_step = {}
        self._type = ""    # TODO: see fime below. # "onetoone" or "regular"

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
    # FIXME: The type can be deftermine when creating the NodeConfigs, 
    # Consider removing this property and setter form here, and make it a responsibilbility of the inputs parser (see PR)
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

    def get_outgoing_link_to(self, down_node_id):
        """Return the outgoing link from this node to a downstream node id."""
        for link in self.outgoing_links:
            if link.end_node is not None and link.end_node.node_id == down_node_id:
                return link
        return None

        # Path finder attributes — pre-initialized to avoid hasattr checks in hot path
        self.node_turn_probs = {}  # {(o,d): {(up_node, down_node): prob}}
        self.turns_distances = {}  # {(o,d): {up_node: {down_node: distance}}}
        self.up_od_probs = defaultdict(lambda: defaultdict(int))  # {up_node: {od_pair: prob}}
        self.current_turn_probs_step = {}  # {od_pair: last_computed_timestep}

    def create_virtual_links(self, params: dict, origin_nodes: list)-> None:
        """Creates a pair virtual links for origin and destination nodes.Virtual links will be attached to the node's incoming and outgoing link lists.

        Args:
            params (dict): A dictionary containing configuration parameters virtual link parameters, including 'simulation_steps'.
            origin_nodes (list): A list of node IDs that are considered origin nodes.
        
        Returns:
           None.
        """
        # TODO: can this be move one step up, to network creation?
        from pednstream.ltm.link import VirtualLinkCreator
        # Value pairs for incoming and outgoing virtual links
        directions = [("in", True), ("out", False) ]

        # Create vitural links
        for direction, is_incoming in directions:
            # create a link, and mark it as vitual
            v_link = VirtualLinkCreator().create_link(params)
            v_link.direction = direction

            if is_incoming:
                # TODO: Reference over Id for end/start nodes. 
                # Should the end_node be a reference to the node object or just the id? 
                v_link.end_node = self.id
                self.incoming_links.append(v_link)
                self.virtual_incoming_link = v_link
            else:
                v_link.start_node = self.id
                self.outgoing_links.append(v_link)
                self.virtual_outgoing_link = v_link

        # Attach demand generator to node:
        if self.id in origin_nodes and self.generator is not None:
            demand_config = params.get('demand', {}).get(f"origin_{self.id}", {})

            pattern = demand_config.get('pattern', 'gaussian_peaks')

            self._init_demand_generator(params['simulation_steps'], params)
            self.demand = self.generator.generate_custom(self.id, pattern)
            # FIXME: implement logger for the following. 
            # if self.logger and self.verbose:
            #     self.logger.info(
            #         f"Total demand of origin node {node.node_id}: {np.sum(node.demand)}"
            #     )
        else:
            self.demand = np.zeros(params['simulation_steps'])

        return None
    
    def _init_demand_generator(self, simulation_steps: int, config: dict) -> None:
        """Creates a demand generator for the node, if it is an origin node.

        Args:
            simulation_steps (int): The number of simulation steps for which to generate demand.
            config (dict): Configuration parameters for the demand generator.
       
        Returns:
            None.
        """
        import logging
        self.generator = DemandGenerator(simulation_steps=simulation_steps, params=config, logger=logging.getLogger('VirtualLink'))
        return None

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

    def update_node_turn_probs(
        self, od_pair, time_step, alpha, beta, omega, temp, std_dev
    ):
        """Update turn probabilities for this node, P(down|up,od)."""
        # Reuse the probabilities within the same timestep so one choice set
        # is generated from one consistent utility/noise realization.
        if self.current_turn_probs_step.get(od_pair) == time_step:
            return self.node_turn_probs

        for up_node, down_nodes in self.turns_distances[od_pair].items():
            if down_nodes:
                turns = list((up_node, down_node) for down_node in down_nodes)
                distances = list(down_nodes.values())
                densities = []
                capacities = []
                for down_node in down_nodes:
                    link = self.get_outgoing_link_to(down_node)
                    if link is not None:
                        densities.append(
                            np.maximum(
                                link.get_density(max(0, time_step - 1))
                                - link.k_critical,
                                0,
                            )
                            / (link.k_jam - link.k_critical)
                        )
                        capacity = link.receiving_flow[
                            max(0, time_step - 2)
                        ]  # -2 steps is the most recent, the capacity of -1 step is -1 by default
                        capacities.append(
                            capacity
                            if capacity >= 0
                            else link.back_gate_width
                            * link.free_flow_speed
                            * link.k_critical
                            * link.unit_time
                        )
                    else:
                        # this is the case of origin/destination nodes, with down node id -1
                        densities.append(0)
                        capacities.append(100)

                norm_densities = np.array(densities)
                if std_dev == 0:
                    time_variation = 0
                else:
                    time_variation = np.random.normal(0, std_dev, len(turns))
                utilities = (
                    alpha * np.array(distances) / (np.sum(distances) + 1e-6)
                    + beta * norm_densities
                    - omega * np.array(capacities) / (np.sum(capacities) + 1e-6)
                ) + time_variation
                exp_utilities = np.exp(-temp * utilities)
                probs = exp_utilities / np.sum(exp_utilities)
                self.node_turn_probs[od_pair].update(dict(zip(turns, probs)))

        self.current_turn_probs_step[od_pair] = time_step
        return self.node_turn_probs

    def update_turning_fractions(
        self, time_step: int, od_manager, alpha, beta, omega, temp, std_dev
    ):
        """Calculate turning fractions using stored turn probabilities."""
        turning_fractions = np.zeros(self.edge_num)
        # Update P(od|up) for each upstream node
        for up_node, od_pairs in self.up_od_probs.items():
            total_flow = 0
            # First pass: calculate total flow
            for od_pair in od_pairs:
                flow = od_manager.get_od_flow(od_pair[0], od_pair[1], time_step)
                od_pairs[od_pair] = flow
                total_flow += flow

            # Second pass: normalize to get probabilities
            if total_flow > 0:
                for od_pair in od_pairs:
                    od_pairs[od_pair] /= total_flow
            else:
                # If no flow, set equal probabilities
                n_pairs = len(od_pairs)
                for od_pair in od_pairs:
                    od_pairs[od_pair] = 1.0 / n_pairs if n_pairs > 0 else 0

        # Calculate final turning fractions
        upstream_nodes = [
            link.start_node.node_id if link.start_node is not None else -1
            for link in self.incoming_links
        ]
        downstream_nodes = [
            link.end_node.node_id if link.end_node is not None else -1
            for link in self.outgoing_links
        ]

        idx = 0
        for up in upstream_nodes:
            for down in downstream_nodes:
                if up == down:
                    continue
                turn = (up, down)
                prob_sum = 0

                # Get all OD pairs for this turn
                od_pairs = self.ods_in_turns.get(turn, set())
                for od_pair in od_pairs:
                    # P(down|up,od) from node_turn_probs
                    self.update_node_turn_probs(
                        od_pair,
                        time_step=time_step,
                        alpha=alpha,
                        beta=beta,
                        omega=omega,
                        temp=temp,
                        std_dev=std_dev,
                    )
                    turn_prob = self.node_turn_probs[od_pair].get(turn, 0)
                    # P(od|up) from up_od_probs
                    od_prob = self.up_od_probs[up].get(od_pair, 0)
                    prob_sum += turn_prob * od_prob

                turning_fractions[idx] = prob_sum
                idx += 1

        return turning_fractions

    def check_fractions(self):
        """
        Check if the turning fractions are valid and normalize if needed.

        Normalization strategy:
        - If sum > 0: normalize by dividing by sum (preserves relative magnitudes)
        - If sum == 0: fall back to equal probabilities (no information available)
        """
        fract = self.turning_fractions.reshape(self.dest_num, self.source_num - 1)

        for i in range(self.dest_num):
            row_sum = np.sum(fract[i])

            if np.abs(row_sum - 1) > 1e-3:
                if row_sum > 1e-6:
                    # Normalize by sum - preserves relative probabilities
                    fract[i] = fract[i] / row_sum
                    print(
                        f"Warning: turning fractions at node {self.node_id} for downstream {i} do not sum to 1. Normalizing."
                    )
                else:
                    # No information available - use equal probabilities
                    fract[i] = np.ones(self.source_num - 1) / (self.source_num - 1)

        self.turning_fractions = fract.flatten()
        return self.turning_fractions

    def calculate_node_turning_fractions(
        self, time_step: int, od_manager, alpha, beta, omega, temp, std_dev
    ):
        """Calculate and store this node's current turning fractions."""
        if self.source_num > 2:
            fractions = self.update_turning_fractions(
                time_step, od_manager, alpha, beta, omega, temp, std_dev
            )
            self.turning_fractions = fractions
            self.check_fractions()

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
