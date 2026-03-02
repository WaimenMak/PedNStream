# -*- coding: utf-8 -*-
# @Time    : 26/01/2025 15:30
# @Author  : mmai
# @FileName: builders
# @Software: PyCharm

"""
Observation builders and action appliers for PedNet multi-agent environment.

Handles conversion between agent actions/observations and network state.

Observation Modes:
    - "flow_only": Only flow features (inflow, outflow) - minimal observation
    - "with_gate": Flow features + gate width (for gater agents)
    - "with_density": Flow + density features
    - "full": All available features (density, flow, gate width)
"""

import numpy as np
from typing import Dict, Any, List
from .discovery import AgentManager
# from src.LTM.link import Link, Separator
from pednstream.ltm.link import Link, Separator


class ObservationBuilder:
    """
    Builds observations for separator and gater agents from network state.
    """
    
    def __init__(self, network, agent_manager: AgentManager, normalize: bool = False, obs_mode: str = "flow_only"):
        """
        Initialize observation builder.
        
        Args:
            network: Network instance from LTM simulation
            agent_manager: AgentManager instance with agent mappings
            normalize: Whether to normalize observations
            obs_mode: Observation mode - one of: "option1", "option2", "option3", "option4"
        """
        self.network = network
        self.agent_manager = agent_manager
        self.normalize = normalize
        self.obs_mode = obs_mode
        # Validate obs_mode
        valid_modes = ["option1", "option2", "option3", "option4", "option5"]
        if obs_mode not in valid_modes:
            raise ValueError(f"obs_mode must be one of {valid_modes}, got: {obs_mode}")
        # Determine features per link based on obs_mode for gater
        # Each option now includes both front_gate_width and back_gate_width
        if self.obs_mode == "option1":
            self.features_per_link = 4  # inflow, reverse outflow, front_gate, back_gate
        elif self.obs_mode == "option2":
            self.features_per_link = 5  # inflow, reverse outflow, density, front_gate, back_gate
        elif self.obs_mode == "option3":
            self.features_per_link = 6  # inoutflow, reverse inoutflow, front_gate, back_gate
        elif self.obs_mode == "option4":
            self.features_per_link = 7  # inoutflow, reverse inoutflow, density, front_gate, back_gate
        elif self.obs_mode == "option5":
            self.features_per_link = 6  # travel_time_ratio, density_ratio, demand, throughput, back_gate, rev_back_gate
        else:
            raise ValueError(f"Unknown observation mode: {self.obs_mode}")

        # Normalization constants (will be refined based on actual features)
        self.density_norm = 6.0    # Typical jam density
        self.speed_norm = 1.5       # Typical free-flow speed
        # self.time_norm = 100.0      # Typical travel time
        self.flow_norm = 20.0       # Typical flow rate (reduced from 10.0 to amplify signal)
        # self.link_widths = []      # list of link widths for normalization
        self.unit_time = network.params['unit_time']
    
    def build_observation(self, agent_id: str, time_step: int) -> np.ndarray:
        """
        Build observation for given agent at current time step.
        
        Args:
            agent_id: Agent identifier
            time_step: Current simulation time step
            
        Returns:
            Observation array for the agent
        """
        agent_type = self.agent_manager.get_agent_type(agent_id)
        
        if agent_type == "sep":
            return self._build_separator_observation(agent_id, time_step)
        elif agent_type == "gate":
            return self._build_gater_observation(agent_id, time_step)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def _build_separator_observation(self, agent_id: str, time_step: int) -> np.ndarray:
        """Build observation for separator agent."""
        forward_link, reverse_link = self.agent_manager.get_separator_links(agent_id)
        
        features = []
        
        # Forward direction features
        # if self.with_density_obs:
        #     features.extend([
        #         forward_link.inflow[time_step] if time_step < len(forward_link.inflow) else 0.0,
        #         forward_link.outflow[time_step] if time_step < len(forward_link.outflow) else 0.0, # outflow of the forward link
        #         reverse_link.density[time_step] if time_step < len(reverse_link.density) else 0.0, # density of the reverse link
        #         reverse_link.inflow[time_step] if time_step < len(reverse_link.inflow) else 0.0, # inflow of the reverse link
        #         reverse_link.outflow[time_step] if time_step < len(reverse_link.outflow) else 0.0, # outflow of the reverse link
        #     ])
        # else:
        features.extend([
            forward_link.inflow[time_step] if time_step < len(forward_link.inflow) else 0.0,
            forward_link.outflow[time_step] if time_step < len(forward_link.outflow) else 0.0,
            reverse_link.inflow[time_step] if time_step < len(reverse_link.inflow) else 0.0,
            reverse_link.outflow[time_step] if time_step < len(reverse_link.outflow) else 0.0,
        ])
        
        obs = np.array(features, dtype=np.float32)
        
        # Apply normalization if enabled
        if self.normalize:
            obs = self._normalize_separator_obs(obs)
        
        return obs
    
    def _build_gater_observation(self, agent_id: str, time_step: int) -> np.ndarray:
        """Build observation for gater agent."""
        out_links = self.agent_manager.get_gater_outgoing_links(agent_id)
        max_outdegree = self.agent_manager.get_max_outdegree(agent_id)  # no padding
        
        obs = np.zeros(max_outdegree * self.features_per_link, dtype=np.float32)
        current_link_widths = []

        for i, link in enumerate(out_links):
            start_idx = i * self.features_per_link
            
            # Extract link features based on obs_mode
            # All options now include back_gate_width (for the outgoing link) 
            # and reverse_link.front_gate_width (for the incoming link)
            if self.obs_mode == "option1":
                link_features = [
                    link.inflow[time_step] if time_step < len(link.inflow) else 0.0,
                    link.reverse_link.outflow[time_step] if time_step < len(link.reverse_link.outflow) else 0.0,
                    link.back_gate_width,
                    link.reverse_link.front_gate_width,
                ]
            elif self.obs_mode == "option2":
                link_features = [
                    link.inflow[time_step] if time_step < len(link.inflow) else 0.0,
                    link.reverse_link.outflow[time_step] if time_step < len(link.reverse_link.outflow) else 0.0,
                    link.get_density(time_step), # shared density
                    link.back_gate_width,
                    link.reverse_link.front_gate_width,
                ]
            elif self.obs_mode == "option3":
                link_features = [
                    link.inflow[time_step] if time_step < len(link.inflow) else 0.0,
                    link.outflow[time_step] if time_step < len(link.outflow) else 0.0,
                    link.reverse_link.inflow[time_step] if time_step < len(link.reverse_link.inflow) else 0.0,
                    link.reverse_link.outflow[time_step] if time_step < len(link.reverse_link.outflow) else 0.0,
                    link.back_gate_width,
                    # link.reverse_link.front_gate_width,
                    link.reverse_link.back_gate_width,
                ]
                current_link_widths.append(link.width)

            elif self.obs_mode == "option4":
                link_features = [
                    link.inflow[time_step] if time_step < len(link.inflow) else 0.0,
                    link.outflow[time_step] if time_step < len(link.outflow) else 0.0,
                    link.reverse_link.inflow[time_step] if time_step < len(link.reverse_link.inflow) else 0.0,
                    link.reverse_link.outflow[time_step] if time_step < len(link.reverse_link.outflow) else 0.0,
                    link.get_density(time_step) if time_step < len(link.density) else 0.0,
                    link.back_gate_width,
                    # link.reverse_link.front_gate_width,
                    link.reverse_link.back_gate_width,
                ]
            elif self.obs_mode == "option5":
                # 1. Normalized Travel Time
                T_free = link.length / link.free_flow_speed if link.free_flow_speed > 0 else 1.0
                tt = link.travel_time[time_step] if time_step < len(link.travel_time) else T_free
                
                # 2. Critical Density Ratio
                k_crit = link.k_critical if hasattr(link, 'k_critical') else 2.0
                density_ratio = link.get_density(time_step) / k_crit
                
                link_features = [
                    tt / T_free,
                    density_ratio,
                    link.reverse_link.inflow[time_step] if time_step < len(link.reverse_link.inflow) else 0.0,
                    link.outflow[time_step] if time_step < len(link.outflow) else 0.0,
                    link.back_gate_width,
                    link.reverse_link.back_gate_width,
                ]
            
            obs[start_idx:start_idx + self.features_per_link] = link_features
        
        # Apply normalization if enabled
        if self.normalize:
            obs = self._normalize_gater_obs(obs, current_link_widths)
        
        return obs
    
    def _normalize_separator_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize separator observation features based on obs_mode."""
        normalized = obs.copy()
        
        # if self.obs_mode == "option1":
        #     # All flow features (indices 0-3)
        #     normalized[:] /= self.flow_norm
        # elif self.obs_mode == "option2":
        #     # Flow features (indices 0-3), separator width not normalized (index 4)
        #     normalized[:4] /= self.flow_norm
        # elif self.obs_mode == "option3":
        #     # Density features (indices 0, 3)
        #     # normalized[[0, 3]] /= self.density_norm
        #     # Flow features (indices 1, 2, 4, 5)
        #     normalized[[0, 1, 2, 3]] /= self.flow_norm
        # elif self.obs_mode == "option4":
        #     # Density features (indices 0, 3)
        #     normalized[[0, 3]] /= self.density_norm
        #     # Flow features (indices 1, 2, 4, 5)
        #     normalized[[1, 2, 4, 5]] /= self.flow_norm
        #     # Separator width not normalized (index 6)
        
        return normalized
    
    def _normalize_gater_obs(self, obs: np.ndarray, link_widths: List[float] = None) -> np.ndarray:
        """Normalize gater observation features based on obs_mode."""
        normalized = obs.copy()
        
        
        if self.features_per_link == 0:
            return normalized

        max_outdegree = len(obs) // self.features_per_link
        
        for i in range(max_outdegree):
            start_idx = i * self.features_per_link
            
            if self.obs_mode == "option1":
                # Normalize both flow features (indices 0, 1)
                normalized[start_idx] /= self.flow_norm
                normalized[start_idx + 1] /= self.flow_norm
            elif self.obs_mode == "option2":
                # Normalize flow (indices 0, 1), gate width not normalized (index 2)
                normalized[start_idx] /= self.flow_norm
                normalized[start_idx + 1] /= self.flow_norm
            elif self.obs_mode == "option3":
                if link_widths and i < len(link_widths):
                    width = link_widths[i]
                    # Max flow estimate: reduced scale to amplify the observation signal
                    # max_flow = width * self.unit_time * 1.0
                    max_flow = self.flow_norm
                    
                    # Normalize flows (indices 0-3)
                    normalized[start_idx] = normalized[start_idx]/max_flow
                    normalized[start_idx + 1] = normalized[start_idx + 1]/max_flow
                    normalized[start_idx + 2] = normalized[start_idx + 2]/max_flow
                    normalized[start_idx + 3] = normalized[start_idx + 3]/max_flow
                    # Normalize front_gate_width (index 4) and back_gate_width (index 5)
                    # normalized[start_idx + 4] = np.clip(normalized[start_idx + 4]/width, 0, 1)
                    # normalized[start_idx + 5] = np.clip(normalized[start_idx + 5]/width, 0, 1)
            elif self.obs_mode == "option4":
                max_flow = self.flow_norm
                
                # Normalize flows (indices 0-3)
                normalized[start_idx] = normalized[start_idx]/max_flow
                normalized[start_idx + 1] = normalized[start_idx + 1]/max_flow
                normalized[start_idx + 2] = normalized[start_idx + 2]/max_flow
                normalized[start_idx + 3] = normalized[start_idx + 3]/max_flow
            elif self.obs_mode == "option5":
                max_flow = self.flow_norm
                
                # tt_ratio (0) and density_ratio (1) are naturally bounded/interpretable
                # We normalize demand (idx 2) and throughput (idx 3) using flow_norm
                normalized[start_idx + 2] = normalized[start_idx + 2]/max_flow
                normalized[start_idx + 3] = normalized[start_idx + 3]/max_flow
        
        return normalized


class ActionApplier:
    """
    Applies agent actions to network components.
    """
    
    def __init__(self, network, agent_manager: AgentManager, 
                 max_delta_sep_width: float = 0.1, max_delta_gate_width: float = 0.1, min_sep_width: float = 1.0):
        """
        Initialize action applier. for all kinds of algorithms, not just RL.
        
        Args:
            network: Network instance from LTM simulation
            agent_manager: AgentManager instance with agent mappings
            max_delta_sep_width: Maximum delta separator width within one time step
            max_delta_gate_width: Maximum delta gate width within one time step
            min_sep_width: Minimum separator width
        """
        self.network = network
        self.agent_manager = agent_manager
        self.max_delta_sep_width = max_delta_sep_width
        self.max_delta_gate_width = max_delta_gate_width
        self.min_sep_width = min_sep_width
    
    def apply_all_actions(self, actions: Dict[str, Any]):
        """
        Apply all agent actions to the network.
        
        Args:
            actions: Dictionary mapping agent_id to action
        """
        for agent_id, action in actions.items():
            agent_type = self.agent_manager.get_agent_type(agent_id)
            
            if agent_type == "sep":
                self._apply_separator_action(agent_id, action)
            elif agent_type == "gate":
                self._apply_gater_action(agent_id, action)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
    
    def clip_separator_action_value(self, action_value: float, forward_link: Link):
        """
        Validate separator agent action value. if the action value is >=1 meter and <= max link width.
        If the change of the width is too large, clip the action value to the maximum or minimum value.
        """
        # if action_value < self.min_sep_width or action_value > forward_link.width - self.min_sep_width:
        #     return np.clip(action_value, self.min_sep_width, forward_link.width - self.min_sep_width)
        if abs(action_value - forward_link.separator_width) > self.max_delta_sep_width:
            delta = np.clip(
                action_value - forward_link.separator_width,
                -self.max_delta_sep_width,
                self.max_delta_sep_width
            )
            action_value = forward_link.separator_width + delta
        return np.clip(action_value, self.min_sep_width, forward_link.width - self.min_sep_width)

    def clip_gater_back_gate_action(self, action_value: float, link: Link):
        """
        Clip back gate action value to [0, link.width] with max delta constraint.
        """
        if abs(action_value - link.back_gate_width) > self.max_delta_gate_width:
            delta = np.clip(
                action_value - link.back_gate_width,
                -self.max_delta_gate_width,
                self.max_delta_gate_width
            )
            action_value = link.back_gate_width + delta
        return np.clip(action_value, 0.0, link.width)

    def clip_gater_front_gate_action(self, action_value: float, link: Link):
        """
        Clip front gate action value to [0, link.width] with max delta constraint.
        """
        if abs(action_value - link.front_gate_width) > self.max_delta_gate_width:
            delta = np.clip(
                action_value - link.front_gate_width,
                -self.max_delta_gate_width,
                self.max_delta_gate_width
            )
            action_value = link.front_gate_width + delta
        return np.clip(action_value, 0.0, link.width)

    # Keep old name as alias for backward compatibility
    def clip_gater_action_value(self, action_value: float, link: Link):
        return self.clip_gater_back_gate_action(action_value, link)
    
    def _apply_separator_action(self, agent_id: str, action: np.ndarray):
        """
        Apply separator agent action to control lane allocation.
        
        Args:
            agent_id: Separator agent identifier
            action: Action array of shape (1,) is the processed action value: the actual width
        """
        # Get separator links
        forward_link, reverse_link = self.agent_manager.get_separator_links(agent_id)
        
        # Convert action to width fraction, ensuring minimum width
        action_value = float(action[0]) # should be the actual width of the forward link
        action_value = self.clip_separator_action_value(action_value, forward_link)
        
        # Set separator width (reverse width is automatically adjusted)
        forward_link.separator_width = action_value
    
    def _apply_gater_action(self, agent_id: str, action: np.ndarray):
        """
        Apply gater agent action to control gate widths.
        
        Args:
            agent_id: Gater agent identifier
            action: Action array of shape (num_outgoing_links * 2,)
                    Layout: [front_gate_0, back_gate_0, front_gate_1, back_gate_1, ...]
        """
        # Get gater outgoing links
        out_links = self.agent_manager.get_gater_outgoing_links(agent_id)
        
        # Apply actions to all outgoing links (2 actions per link: front + back)
        for i, link in enumerate(out_links):
            # Action value is the actual width of the gate
            back_action = float(action[i * 2])
            back_action = self.clip_gater_action_value(back_action, link)
            link.back_gate_width = back_action

            # front_action = float(action[i * 2 + 1])
            # front_action = self.clip_gater_front_gate_action(front_action, link.reverse_link)
            # link.reverse_link.front_gate_width = front_action

            rev_back_action = float(action[i * 2 + 1])
            rev_back_action = self.clip_gater_back_gate_action(rev_back_action, link.reverse_link)
            link.reverse_link.back_gate_width = rev_back_action

