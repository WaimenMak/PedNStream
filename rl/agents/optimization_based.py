# -*- coding: utf-8 -*-
# @Time    : 16/01/2026 14:30
# @Author  : mmai
# @FileName: optimization_based
# @Software: PyCharm

"""
Optimization-based baseline agent for crowd control using mathematical model.

This agent solves a constrained optimization problem at each time step to minimize
the variance of pedestrian density across the network by controlling gate widths.

Mathematical Model:
- Decision Variable: w_i ∈ [0, w_max] (gate width)
- Route Choice: p(l_i | l_k, d; w) = exp(U_kid(w_i)) / Σ_m exp(U_kmd(w_m))
- Utility: U = β_1 * distance + β_2 * density + β_3 * gate_width + ε
- Flow: f_ki(w) = C_k * p(l_i | l_k; w), where C_k = w_k * F_k
- State Update: N_i(t+1; w) = N_i(t) - Σ_j w_i F_i p(l_j|l_i; w) + Σ_k w_k F_k p(l_i|l_k; w)
- Objective: min_w Var(N(t+1; w))
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
from scipy.optimize import differential_evolution, Bounds


class OptimizationBasedAgent(ABC):
    """
    Base class for optimization-based agents that solve a mathematical model
    to determine optimal gate widths.
    """
    
    def __init__(self, network, agent_manager, verbose: bool = False):
        """
        Initialize the optimization-based agent.
        
        Args:
            network: The LTM network instance
            agent_manager: AgentManager instance for accessing network topology
            verbose: Whether to print optimization details
        """
        self.network = network
        self.agent_manager = agent_manager
        self.verbose = verbose
        
        # Extract model parameters from PathFinder
        path_finder = network.path_finder
        self.temp = path_finder.temp if path_finder else 0.1
        self.alpha = path_finder.alpha if path_finder else 1.0  # distance weight
        self.beta_density = path_finder.beta if path_finder else 0.05  # congestion weight
        # β_3: gate width utility weight (positive = wider gates are more attractive)
        self.beta_width = path_finder.omega  # Tunable parameter
        
        # Cache network topology
        self._build_topology_cache()
        
    def _build_topology_cache(self):
        """Build cached topology data structures for fast computation."""
        pass
    
    def take_action(self, obs: Dict[str, np.ndarray], time_step: int = None) -> Dict[str, np.ndarray]:
        """
        Solve optimization problem and return actions for all agents.
        
        Args:
            obs: Dictionary of observations for all agents
            time_step: Current simulation time step (required for accessing history)
            
        Returns:
            Dictionary of actions (gate widths) for all agents
        """
        pass


# class CentralizedOptimizationAgent(OptimizationBasedAgent):
#     """
#     Centralized optimization agent that jointly optimizes all gate widths
#     to minimize network-wide density variance.
#     """
    
#     def __init__(self, network, agent_manager, verbose: bool = False):
#         super().__init__(network, agent_manager, verbose)
        
#     def _build_topology_cache(self):
#         """Build cached topology for centralized optimization."""
#         # Get all gater agents (we focus on gate control, not separators for now)
#         self.gater_agent_ids = list(self.agent_manager.get_gater_agents().keys())
#         self.n_agents = len(self.gater_agent_ids)
        
#         # Map agent_id -> list of controlled links
#         self.agent_to_links = {}
#         self.agent_to_node = {}
#         for agent_id in self.gater_agent_ids:
#             node = self.agent_manager.get_gater_node(agent_id)
#             out_links = self.agent_manager.get_gater_outgoing_links(agent_id)
#             self.agent_to_links[agent_id] = out_links
#             self.agent_to_node[agent_id] = node
        
#         # Build decision variable mapping
#         # Decision variables: one per controlled outgoing link
#         self.decision_vars = []  # List of (agent_id, link_idx, link)
#         for agent_id in self.gater_agent_ids:
#             for link_idx, link in enumerate(self.agent_to_links[agent_id]):
#                 self.decision_vars.append((agent_id, link_idx, link))
        
#         self.n_decision_vars = len(self.decision_vars)
        
#         # Bounds: [0, link.width] for each decision variable
#         self.bounds = [(0, link.width) for _, _, link in self.decision_vars]
        
#         # Build state vector: all links in the network
#         self.all_links = list(self.network.links.values())
#         self.n_links = len(self.all_links)
#         self.link_to_idx = {(link.start_node.node_id if link.start_node else -1,
#                              link.end_node.node_id if link.end_node else -1): idx
#                             for idx, link in enumerate(self.all_links)}
        
#     def _get_current_state(self, time_step: int = None) -> np.ndarray:
#         """
#         Extract current network state N(t).
        
#         Args:
#             time_step: Current simulation step. If None, uses last recorded step.
            
#         Returns:
#             np.ndarray: Current occupancy for each link
#         """
#         N_t = np.zeros(self.n_links)
#         idx_t = time_step - 1 if time_step is not None and time_step > 0 else -1
        
#         for idx, link in enumerate(self.all_links):
#             # Get current number of pedestrians (at beginning of step t, which is end of t-1)
#             # If using negative index, careful with 0 initialization
#             if idx_t >= 0 and idx_t < len(link.num_pedestrians):
#                  N_t[idx] = link.num_pedestrians[idx_t]
#             elif len(link.num_pedestrians) > 0:
#                  N_t[idx] = link.num_pedestrians[idx_t] # Fallback to -1
#             else:
#                  N_t[idx] = 0.0
                 
#         return N_t
    
#     def _compute_route_choice_probs(self, w_vector: np.ndarray, node, time_step: int = None) -> Dict:
#         """
#         Compute route choice probabilities p(l_i | l_k, d; w) for a given node.
#         """
#         if not hasattr(node, 'node_turn_probs') or not hasattr(node, 'turns_distances'):
#             return {}
        
#         route_probs = {}  # {od_pair: {(up_node, down_node): probability}}
        
#         idx_t = time_step - 1 if time_step is not None and time_step > 0 else -1
        
#         # For each OD pair passing through this node
#         for od_pair in node.node_turn_probs.keys():
#             if od_pair not in node.turns_distances:
#                 continue
                
#             route_probs[od_pair] = {}
            
#             # For each upstream node
#             for up_node, down_nodes in node.turns_distances[od_pair].items():
#                 if not down_nodes:
#                     continue
                
#                 # Collect all data for downstream options first
#                 turns = []
#                 distances = []
#                 densities = []
#                 gate_widths = []
#                 link_widths = []
#                 k_criticals = []
#                 k_jams = []
#                 down_node_ids = []
                
#                 # IMPORTANT: ensure deterministic ordering so arrays align by downstream node id
#                 for down_node, distance in sorted(down_nodes.items(), key=lambda kv: kv[0]):
#                     turn = (up_node, down_node)
#                     turns.append(turn)
#                     distances.append(distance)
#                     down_node_ids.append(down_node)
                    
#                     # Get link for this downstream option
#                     try:
#                         link = self.network.links[(node.node_id, down_node)]
                        
#                         # Get current density
#                         density = 0.0
#                         if idx_t >= 0 and idx_t < len(link.density):
#                             density = link.density[idx_t]
#                         elif len(link.density) > 0:
#                             density = link.density[-1]
#                         densities.append(density)
                        
#                         # Get gate width from decision variables if controlled, else use current
#                         gate_width = link.front_gate_width  # Default to current
#                         for var_idx, (agent_id, link_idx, controlled_link) in enumerate(self.decision_vars):
#                             if controlled_link == link:
#                                 gate_width = w_vector[var_idx]
#                                 break
#                         gate_widths.append(gate_width)
#                         link_widths.append(link.width)
#                         k_criticals.append(link.k_critical)
#                         k_jams.append(link.k_jam)
                        
#                     except KeyError:
#                         # Destination or virtual link
#                         densities.append(0.0)
#                         gate_widths.append(0.0)
#                         link_widths.append(1.0)  # Avoid division by zero
#                         k_criticals.append(2.0)  # Default
#                         k_jams.append(10.0)  # Default
                
#                 # Convert to numpy arrays for vectorized computation
#                 distances = np.array(distances)
#                 densities = np.array(densities)
#                 gate_widths = np.array(gate_widths)
#                 link_widths = np.array(link_widths)
#                 k_criticals = np.array(k_criticals)
#                 k_jams = np.array(k_jams)
                
#                 # Compute utilities vectorized: U = α*distance + β*density + β_3*width
#                 # Normalize for numerical stability
#                 norm_distances = distances / (distances + 1e-6)
#                 norm_densities = np.maximum(densities - k_criticals, 0) / (k_jams - k_criticals + 1e-6)
#                 norm_widths = gate_widths / (link_widths + 1e-6)
                
#                 # Compute all utilities at once
#                 utilities = (self.alpha * norm_distances + 
#                            self.beta_density * norm_densities +
#                            self.beta_width * norm_widths)
                
#                 # Compute logit probabilities
#                 exp_utilities = np.exp(-self.temp * utilities)
#                 probs = exp_utilities / (np.sum(exp_utilities) + 1e-10)
                
#                 # Store probabilities
#                 for turn, prob in zip(turns, probs):
#                     route_probs[od_pair][turn] = prob
        
#         return route_probs
    
#     def _compute_aggregated_route_probs(self, w_vector: np.ndarray, node, time_step: int = None) -> Dict:
#         """
#         Compute aggregated route probabilities p(l_i | l_k; w) by summing over OD pairs.
#         """
#         # Get route choice probabilities per OD
#         route_probs_od = self._compute_route_choice_probs(w_vector, node, time_step)
        
#         if not route_probs_od:
#             return {}
        
#         # Get OD demand splits p(d|l_k)
#         aggregated_probs = {}
        
#         # For each upstream link
#         if hasattr(node, 'up_od_probs'):
#             for up_node, od_dict in node.up_od_probs.items():
#                 # Normalize OD probabilities (they should already be normalized)
#                 total_od_prob = sum(od_dict.values())
#                 if total_od_prob < 1e-10:
#                     continue
                
#                 # For each possible downstream link
#                 downstream_nodes = set()
#                 for od_pair in od_dict.keys():
#                     if od_pair in route_probs_od:
#                         for turn in route_probs_od[od_pair].keys():
#                             if turn[0] == up_node:
#                                 downstream_nodes.add(turn[1])
                
#                 # Aggregate: p(down|up) = Σ_d p(d|up) * p(down|up,d)
#                 for down_node in downstream_nodes:
#                     prob_sum = 0.0
#                     for od_pair, od_prob in od_dict.items():
#                         if od_pair in route_probs_od:
#                             turn = (up_node, down_node)
#                             turn_prob = route_probs_od[od_pair].get(turn, 0.0)
#                             prob_sum += (od_prob / total_od_prob) * turn_prob
                    
#                     aggregated_probs[(up_node, down_node)] = prob_sum
        
#         return aggregated_probs
    
#     def _predict_next_state(self, w_vector: np.ndarray, N_t: np.ndarray, time_step: int = None) -> np.ndarray:
#         """
#         Predict next state N(t+1; w) given decision variables w.
#         """
#         N_next = N_t.copy()
        
#         # For each controlled node, compute flow redistribution
#         for agent_id in self.gater_agent_ids:
#             node = self.agent_to_node[agent_id]
#             out_links = self.agent_to_links[agent_id]
            
#             # Update OD probabilities before computing route probs
#             if hasattr(self.network, 'path_finder') and self.network.path_finder is not None:
#                 if hasattr(self.network, 'od_manager'):
#                     self.network.path_finder.update_turning_fractions(node, time_step, self.network.od_manager)
            
#             # Compute aggregated routing probabilities
#             agg_probs = self._compute_aggregated_route_probs(w_vector, node, time_step)
            
#             if not agg_probs:
#                 continue
            
#             # For each outgoing link from this node
#             for link_idx, link in enumerate(out_links):
#                 # Find decision variable index for this link
#                 var_idx = None
#                 for idx, (aid, lidx, lnk) in enumerate(self.decision_vars):
#                     if aid == agent_id and lidx == link_idx:
#                         var_idx = idx
#                         break
                
#                 if var_idx is None:
#                     continue
                
#                 gate_width = w_vector[var_idx]
                
#                 # Compute capacity: C_k = w_k * F_k
#                 F_k = link.free_flow_speed * link.k_critical  # Flow capacity per unit width
#                 C_k = gate_width * F_k * self.network.unit_time
                
#                 # Get link state index
#                 link_key = (link.start_node.node_id if link.start_node else -1,
#                            link.end_node.node_id if link.end_node else -1)
#                 link_idx_state = self.link_to_idx.get(link_key)
                
#                 if link_idx_state is None:
#                     continue
                
#                 # Outflow from this link: Σ_j w_i F_i p(l_j|l_i; w)
#                 outflow = 0.0
#                 for (up, down), prob in agg_probs.items():
#                     if up == node.node_id:
#                         outflow += C_k * prob
                
#                 N_next[link_idx_state] -= outflow
                
#                 # Inflow to downstream links: Σ_k w_k F_k p(l_i|l_k; w)
#                 for (up, down), prob in agg_probs.items():
#                     if down == link.end_node.node_id if link.end_node else -1:
#                         down_link_key = (node.node_id, down)
#                         down_idx = self.link_to_idx.get(down_link_key)
#                         if down_idx is not None:
#                             N_next[down_idx] += C_k * prob
        
#         return np.maximum(N_next, 0)  # Ensure non-negative
    
#     def _objective_function(self, w_vector: np.ndarray, N_t: np.ndarray, time_step: int = None) -> float:
#         """
#         Objective function: Var(N(t+1; w))
#         """
#         N_next = self._predict_next_state(w_vector, N_t, time_step)
#         variance = np.var(N_next)
        
#         if self.verbose:
#             print(f"  Objective evaluation: Var(N) = {variance:.4f}")
        
#         return variance
    
#     def take_action(self, obs: Dict[str, np.ndarray], time_step: int = None) -> Dict[str, np.ndarray]:
#         """
#         Solve optimization problem and return optimal gate widths.
#         """
#         # Get current state
#         N_t = self._get_current_state(time_step)
        
#         # Initial guess: current gate widths
#         w_init = np.array([link.front_gate_width for _, _, link in self.decision_vars])
        
#         if self.verbose:
#             print(f"\nOptimization at step {self.network.simulation_steps}")
#             print(f"  Initial widths: {w_init}")
#             print(f"  Current state variance: {np.var(N_t):.4f}")
        
#         # Solve optimization problem
#         result = minimize(
#             fun=lambda w: self._objective_function(w, N_t, time_step),
#             x0=w_init,
#             method='SLSQP',
#             bounds=self.bounds,
#             options={'maxiter': 50, 'ftol': 1e-4}
#         )
        
#         if self.verbose:
#             print(f"  Optimization {'succeeded' if result.success else 'failed'}")
#             print(f"  Optimal widths: {result.x}")
#             print(f"  Final objective: {result.fun:.4f}")
        
#         # Convert solution to action dictionary
#         actions = {}
#         for agent_id in self.gater_agent_ids:
#             agent_actions = []
#             for link_idx, link in enumerate(self.agent_to_links[agent_id]):
#                 # Find corresponding decision variable
#                 for var_idx, (aid, lidx, lnk) in enumerate(self.decision_vars):
#                     if aid == agent_id and lidx == link_idx:
#                         agent_actions.append(result.x[var_idx])
#                         break
#             actions[agent_id] = np.array(agent_actions, dtype=np.float32)
        
#         return actions


class DecentralizedOptimizationAgent(OptimizationBasedAgent):
    """
    Decentralized optimization agent where each gater independently optimizes
    its own intersection using the same mathematical model as centralized version.
    
    Each agent solves: min_w Var(N_local(t+1; w)) where N_local includes only
    links connected to its intersection (incoming + outgoing).
    """
    
    def __init__(self, network, agent_manager, agent_id, verbose: bool = False):
        # Set agent_id before calling super().__init__() because _build_topology_cache() needs it
        self.agent_id = agent_id
        super().__init__(network, agent_manager, verbose)
        
    def _build_topology_cache(self):
        """Build cached topology for decentralized optimization."""
        # Get all gater agents
        # self.gater_agent_ids = list(self.agent_manager.get_gater_agents().keys())
        
        # Map agent_id -> controlled links and node
        self.agent_to_links = {}
        self.agent_to_node = {}
        # for agent_id in self.gater_agent_ids:
        node = self.agent_manager.get_gater_node(self.agent_id)
        out_links = self.agent_manager.get_gater_outgoing_links(self.agent_id)
        self.agent_to_links[self.agent_id] = out_links
        self.agent_to_node[self.agent_id] = node
    
    def _compute_route_choice_probs_local(self, w_vector: np.ndarray, node, 
                                          out_links: List, time_step: int = None) -> Dict:
        """
        Compute route choice probabilities p(l_i | l_k, d; w) for a given node.
        Same as centralized version but scoped to local node.
        
        Args:
            w_vector: Decision variable vector (gate widths for this node's outgoing links)
            node: The node to compute probabilities for
            out_links: List of outgoing links for this node
            
        Returns:
            Dictionary with routing probabilities
        """
        if not hasattr(node, 'node_turn_probs') or not hasattr(node, 'turns_distances'):
            return {}
        
        route_probs = {}  # {od_pair: {(up_node, down_node): probability}}
        
        # For each OD pair passing through this node
        for od_pair in node.node_turn_probs.keys():
            if od_pair not in node.turns_distances:
                continue
                
            route_probs[od_pair] = {}
            
            # For each upstream node
            for up_node, down_nodes in node.turns_distances[od_pair].items():
                if not down_nodes:
                    continue
                
                # Collect all data for downstream options first
                turns = []
                distances = []
                densities = []
                capacities = []
                down_node_ids = []
                
                # IMPORTANT: ensure deterministic ordering so arrays align by downstream node id
                for down_node, distance in sorted(down_nodes.items(), key=lambda kv: kv[0]):
                    turn = (up_node, down_node)
                    turns.append(turn)
                    distances.append(distance)
                    down_node_ids.append(down_node)
                    
                    # Get link for this downstream option
                    try:
                        link = self.network.links[(node.node_id, down_node)]
                        
                        # Get current density
                        density = link.get_density(time_step) if time_step is not None else link.get_density(-1)
                        densities.append(density)
                        
                        # Get gate width from decision variables if controlled, else use current
                        gate_width = link.back_gate_width  # Default to current
                        for link_idx, controlled_link in enumerate(out_links):
                            if controlled_link == link:
                                gate_width = w_vector[link_idx]
                                break
                        capacities.append(gate_width * link.free_flow_speed * link.k_critical * self.network.unit_time)

                        
                    except KeyError:
                        # Destination or virtual link
                        densities.append(0.0)
                        capacities.append(100) # set a high capacity for origin/destination nodes
                
                # Convert to numpy arrays for vectorized computation
                distances = np.array(distances)
                densities = np.array(densities)
                capacities = np.array(capacities)

                
                # Compute utilities vectorized: U = α*distance + β*density + β_3*width
                # Normalize for numerical stability
                norm_distances = distances / (np.sum(distances) + 1e-6)
                
                # Normalize density: (density - k_critical) / (k_jam - k_critical)
                # Need to get k_critical and k_jam for each link, aligned with down_node_ids
                k_criticals = []
                k_jams = []
                for down_node in down_node_ids:
                    try:
                        link = self.network.links[(node.node_id, down_node)]
                        k_criticals.append(link.k_critical)
                        k_jams.append(link.k_jam)
                    except KeyError:
                        k_criticals.append(2.0)  # Default
                        k_jams.append(10.0)  # Default
                
                k_criticals = np.array(k_criticals)
                k_jams = np.array(k_jams)
                norm_densities = np.maximum(densities - k_criticals, 0) / (k_jams - k_criticals + 1e-6)
                
                norm_capacities = capacities / (np.sum(capacities) + 1e-6)
                
                # Compute all utilities at once
                utilities = (self.alpha * norm_distances + 
                           self.beta_density * norm_densities -
                           self.beta_width * norm_capacities)
                
                # Compute logit probabilities
                exp_utilities = np.exp(-self.temp * utilities)
                probs = exp_utilities / (np.sum(exp_utilities) + 1e-10)
                
                # Store probabilities
                for turn, prob in zip(turns, probs):
                    route_probs[od_pair][turn] = prob
        
        return route_probs
    
    def _compute_aggregated_route_probs_local(self, w_vector: np.ndarray, node,
                                               out_links: List, time_step: int = None) -> Dict:
        """
        Compute aggregated route probabilities p(l_i | l_k; w) for local node.
        
        Args:
            w_vector: Decision variable vector
            node: The node to compute probabilities for
            out_links: List of outgoing links for this node
            
        Returns:
            Dictionary {(up_node, down_node): aggregated_probability}
        """
        # Get route choice probabilities per OD
        route_probs_od = self._compute_route_choice_probs_local(w_vector, node, out_links, time_step)
        
        if not route_probs_od:
            return {}
        
        # Get OD demand splits p(d|l_k)
        aggregated_probs = {}
        
        # For each upstream link
        if hasattr(node, 'up_od_probs'):
            for up_node, od_dict in node.up_od_probs.items():
                # Normalize OD probabilities
                total_od_prob = sum(od_dict.values())
                if total_od_prob < 1e-10:
                    continue
                
                # For each possible downstream link
                downstream_nodes = set()
                for od_pair in od_dict.keys():
                    if od_pair in route_probs_od:
                        for turn in route_probs_od[od_pair].keys():
                            if turn[0] == up_node:
                                downstream_nodes.add(turn[1])
                
                # Aggregate: p(down|up) = Σ_d p(d|up) * p(down|up,d)
                for down_node in downstream_nodes:
                    prob_sum = 0.0
                    for od_pair, od_prob in od_dict.items():
                        if od_pair in route_probs_od:
                            turn = (up_node, down_node)
                            turn_prob = route_probs_od[od_pair].get(turn, 0.0)
                            prob_sum += (od_prob / total_od_prob) * turn_prob
                    
                    aggregated_probs[(up_node, down_node)] = prob_sum
        
        return aggregated_probs
    
    def _predict_local_next_state(self, w_vector: np.ndarray, node, 
                                    out_links: List, in_links: List, time_step: int = None) -> np.ndarray:
        """
        Predict next state N(t+1) considering Arm-based Gate Controls.
        """
        local_links = in_links + out_links
        
        # --- Helper: Map Link to Gate Width ---
        # We assume w_vector is ordered by arm index. 
        # You need a way to know which link belongs to which arm.
        # Option A: link.arm_id
        # Option B: Map based on index logic if fixed topology
    # --- Helper: Map Link to Gate Width (Topology Based) ---
        def get_gate_capacity(target_link):
            # 1. Identify the 'Neighbor Node' (the other end of the arm)
            # If link starts at center, neighbor is end_node.
            # If link ends at center, neighbor is start_node.
            target_link_idx = local_links.index(target_link)
            w_idx = target_link_idx % len(w_vector)
            target_gate_width = w_vector[w_idx]
            capacity = target_gate_width * target_link.free_flow_speed * target_link.k_critical * self.network.unit_time
            return capacity
        # 1. Initialize State (N_t)
        N_t_local = np.zeros(len(local_links))
        for i, link in enumerate(local_links):
            if time_step is not None and len(link.num_pedestrians) > time_step:
                N_t_local[i] = link.num_pedestrians[time_step]
        
        N_next = N_t_local.copy()
        idx_t = time_step if time_step is not None else -1

        # 2. External Flow (Simulator Boundary Conditions)
        for i, link in enumerate(local_links):
            ext_in, ext_out = 0.0, 0.0
            travel_gap = np.floor(link.length / (link.free_flow_speed * self.network.unit_time))
            if idx_t >= 0 and idx_t < len(link.inflow):
                if idx_t - travel_gap >= 0:
                    ext_in = link.inflow[int(idx_t-travel_gap)]
                else:
                    ext_in = 0.0
                ext_out = link.outflow[idx_t]
                if ext_out > 0:
                    pass

            if link in in_links:
                N_next[i] += ext_in
            else:
                N_next[i] -= ext_out

        # 3. INTERNAL FLOW LOGIC
        
        # Step A: Calculate DEMAND (Sending side)
        # We calculate how much each upstream link WANTS to send to each downstream link.
        # Logic: Demand = min(People Present, Sending Gate Capacity) * Route Probability
        
        agg_probs = self._compute_aggregated_route_probs_local(w_vector, node, out_links, time_step)
        
        # Store requests: {down_link_obj: total_requested_inflow}
        downstream_requests = {link: 0.0 for link in out_links}
        
        # Store individual transfers to execute later: list of (up_idx, down_idx, requested_amount)
        potential_transfers = []

        for up_link in in_links:
            up_idx = local_links.index(up_link)
            
            # Constraint 1: SENDING GATE (The gate on the incoming arm)
            sending_capacity = get_gate_capacity(up_link)
            
            # Total potential outflow from this link (before splitting by route)
            total_sending_potential = min(N_t_local[up_idx], sending_capacity)

            # Now split this potential among destinations
            # We need to find all probabilities originating from this up_link
            # (Assuming you can filter agg_probs for this up_link)
            
            for (u_id, d_id), prob in agg_probs.items():
                # Verify this prob belongs to the current up_link
                if u_id == up_link.start_node.node_id: # or however you identify up_link
                    # Identify downstream link
                    down_link = self.network.links.get((node.node_id, d_id))
                    
                    if down_link and prob > 0:
                        amount = total_sending_potential * prob
                        downstream_requests[down_link] += amount
                        
                        down_idx = local_links.index(down_link)
                        potential_transfers.append({
                            'up_idx': up_idx,
                            'down_idx': down_idx,
                            'amount': amount,
                            'down_link': down_link
                        })

        # Step B: Apply SUPPLY Constraints (Receiving side) and Execute
        # We check if the downstream gate can handle the total incoming demand.
        
        for down_link, total_req in downstream_requests.items():
            if total_req <= 1e-9:
                continue
                
            # Constraint 2: RECEIVING GATE (The gate on the outgoing arm)
            receiving_capacity = get_gate_capacity(down_link)
            
            # Calculate Scaling Factor (The "Traffic Jam" effect)
            # If 100 people want in, but gate fits 50, everyone gets scaled by 0.5
            scale = 1.0
            if total_req > receiving_capacity:
                scale = receiving_capacity / total_req
            
            # Execute transfers for this destination
            for transfer in potential_transfers:
                if transfer['down_link'] == down_link:
                    actual_flow = transfer['amount'] * scale
                    
                    # Update State (Conservation of Mass)
                    N_next[transfer['up_idx']] -= actual_flow
                    N_next[transfer['down_idx']] += actual_flow

        return np.maximum(N_next, 0)



    def _optimize_single_agent(self, agent_id: str, time_step: int = None) -> np.ndarray:
        """
        Optimize gate widths using Differential Evolution (Gradient-Free).
        Handles non-linear/discontinuous logic (min/max/if-else) robustly.
        """
        node = self.agent_to_node[agent_id]
        out_links = self.agent_to_links[agent_id]
        
        # Get incoming links (needed for state prediction context)
        in_links = [link for link in node.incoming_links
                if not (hasattr(link, 'virtual_incoming_link') and 
                        link == link.virtual_incoming_link)]
        
        n_vars = len(out_links)
        if n_vars == 0:
            return np.array([], dtype=np.float32)
        
        # 1. Define Bounds
        # w is optimized as ABSOLUTE WIDTH in meters [0, link.width]
        # This aligns with your physical constraints.
        bounds = [(0.0, float(link.width)) for link in out_links]
        
        # 2. Update OD probabilities (Context setup)
        # if hasattr(self.network, 'path_finder') and self.network.path_finder is not None:
        #     if hasattr(self.network, 'od_manager'):
        #         self.network.path_finder.update_turning_fractions(node, time_step, self.network.od_manager)
        
        # 3. Define Objective Function
        def objective(w):
            # Predict state N(t+1) based on proposed widths w
            # NOTE: logic assumes w order matches out_links order
            N_next = self._predict_local_next_state(w, node, out_links, in_links, time_step)
            
            # Calculate Variance (Minimizing spatial unevenness)
            N_next = N_next.reshape(2, -1).sum(axis=0)
            obj = np.var(N_next)
            # return np.sum(N_next)
            # obj = np.max(N_next)
            return obj
        
        # 4. Run Gradient-Free Optimization
        # We use Differential Evolution because the landscape is jagged (due to min/max logic).
        try:
            result = differential_evolution(
                func=objective,
                bounds=bounds,
                strategy='best1bin', # Standard strategy
                maxiter=10,           # LIMIT ITERATIONS: Keep low for speed (it's a baseline)
                popsize=50,           # LIMIT POPULATION: approx 5*n_vars evals per gen
                mutation=(0.5, 1),
                recombination=0.7,
                tol=0.01,            # Loose tolerance is fine for a 1-step heuristic
                polish=False,        # IMPORTANT: Disable polish (L-BFGS-B) as we have no gradients
                disp=False
            )
            
            optimal_widths = result.x
            
        except Exception as e:
            print(f"Optimization failed for agent {agent_id}: {e}")
            # Fallback: keep current widths if solver crashes
            optimal_widths = np.array([link.front_gate_width for link in out_links])
        
        return optimal_widths.astype(np.float32)      
    
    def take_action(self, obs: Dict[str, np.ndarray], time_step: int = None) -> Dict[str, np.ndarray]:
        """
        Compute optimal actions for each agent independently.
        """
        # actions = {}
        # for agent_id in self.gater_agent_ids:
        action = self._optimize_single_agent(self.agent_id, time_step)

        return action


if __name__ == "__main__":
    from rl.pz_pednet_env import PedNetParallelEnv
    
    # Test with small network
    dataset = "small_network"
    # dataset = "two_coordinators"
    # dataset = "45_intersections"
    env = PedNetParallelEnv(dataset, obs_mode="option2", action_gap=1,
                           render_mode="animate", verbose=True)
    
    # Create optimization-based agent (choose centralized or decentralized)
    use_centralized = False  # Set to True for centralized optimization
    observations, infos = env.reset(seed=42, options={"randomize": False})
    if use_centralized:
        print("Using Centralized Optimization Agent")
        # opt_agent = CentralizedOptimizationAgent(
        #     network=env.network,
        #     agent_manager=env.agent_manager,
        #     verbose=True
        # )
    else:
        print("Using Decentralized Optimization Agent")
        opt_agents = {}
        for agent_id in env.agent_manager.get_gater_agents():
            opt_agents[agent_id] = DecentralizedOptimizationAgent(
                network=env.network,
                agent_manager=env.agent_manager,
                agent_id=agent_id,
                verbose=False
            )
    
    # Run simulation
    episode_rewards = {agent_id: 0.0 for agent_id in env.agents}
    
    done = False
    step = 0
    max_steps = 200  # Limit for testing
    
    while not done:
        # Get actions from optimization agent
        actions = {}
        for agent_id in env.agent_manager.get_gater_agents():
            actions[agent_id] = opt_agents[agent_id].take_action(observations, time_step=env.sim_step-1)
        
        # Step environment
        observations, rewards, terminations, truncations, infos = env.step(actions)
        done = any(terminations.values()) or any(truncations.values())
        
        # Accumulate rewards
        for agent_id in env.agents:
            episode_rewards[agent_id] += rewards[agent_id]
        
        step += 1
        if step % 20 == 0:
            print(f"Step {step}/{max_steps} completed")
    
    # Print final results
    print("\n=== Final Results ===")
    for agent_id in env.possible_agents:
        print(f"Agent {agent_id} total reward: {episode_rewards[agent_id]:.2f}")
    
    avg_reward = np.mean(list(episode_rewards.values()))
    print(f"Average reward: {avg_reward:.2f}")
    
    # Save and render
    output_dir = "decentralized_opt" if not use_centralized else "centralized_opt"
    env.save(simulation_dir=f"../../outputs/{output_dir}")
    env.render(simulation_dir=f"../../outputs/{output_dir}", 
              variable='density', vis_actions=True)
