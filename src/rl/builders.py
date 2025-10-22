# -*- coding: utf-8 -*-
# @Time    : 26/01/2025 15:30
# @Author  : mmai
# @FileName: builders
# @Software: PyCharm

"""
Observation builders and action appliers for PedNet multi-agent environment.

Handles conversion between agent actions/observations and network state.
"""

import numpy as np
from typing import Dict, Any, List
from .discovery import AgentDiscovery
from LTM.link import Link, Separator


class ObservationBuilder:
    """
    Builds observations for separator and gater agents from network state.
    """
    
    def __init__(self, network, agent_discovery: AgentDiscovery, normalize: bool = True, with_density_obs: bool = False):
        """
        Initialize observation builder.
        
        Args:
            network: Network instance from LTM simulation
            agent_discovery: AgentDiscovery instance with agent mappings
            normalize: Whether to normalize observations
            with_density_obs: Whether to include density observations
        """
        self.network = network
        self.agent_discovery = agent_discovery
        self.normalize = normalize
        self.with_density_obs = with_density_obs
        
        # Normalization constants (will be refined based on actual features)
        self.density_norm = 10.0    # Typical jam density
        self.speed_norm = 2.0       # Typical free-flow speed
        self.time_norm = 100.0      # Typical travel time
        self.flow_norm = 50.0       # Typical flow rate
    
    def build_observation(self, agent_id: str, time_step: int) -> np.ndarray:
        """
        Build observation for given agent at current time step.
        
        Args:
            agent_id: Agent identifier
            time_step: Current simulation time step
            
        Returns:
            Observation array for the agent
        """
        agent_type = self.agent_discovery.get_agent_type(agent_id)
        
        if agent_type == "sep":
            return self._build_separator_observation(agent_id, time_step)
        elif agent_type == "gate":
            return self._build_gater_observation(agent_id, time_step)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def _build_separator_observation(self, agent_id: str, time_step: int) -> np.ndarray:
        """Build observation for separator agent."""
        forward_link, reverse_link = self.agent_discovery.get_separator_links(agent_id)
        
        # TODO: Extract meaningful features from both directions
        # Placeholder features:
        features = []
        
        # Forward direction features
        if self.with_density_obs:
            features.extend([
                forward_link.density[time_step] if time_step < len(forward_link.density) else 0.0,
                forward_link.inflow[time_step] if time_step < len(forward_link.inflow) else 0.0,
                forward_link.outflow[time_step] if time_step < len(forward_link.outflow) else 0.0,
                reverse_link.density[time_step] if time_step < len(reverse_link.density) else 0.0,
                reverse_link.inflow[time_step] if time_step < len(reverse_link.inflow) else 0.0,
                reverse_link.outflow[time_step] if time_step < len(reverse_link.outflow) else 0.0,
            ])
        else:
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
        # node = self.agent_discovery.get_gater_node(agent_id)
        out_links = self.agent_discovery.get_gater_outgoing_links(agent_id)
        max_outdegree = self.agent_discovery.get_max_outdegree()
        
        # TODO: Extract per-outgoing-link features
        # Placeholder: 8 features per link, padded to max_outdegree
        features_per_link = 3 if self.with_density_obs else 2 # density, inflow of outgoing link, outflow of incoming link
        obs = np.zeros(max_outdegree * features_per_link, dtype=np.float32)
        
        for i, link in enumerate(out_links):
            start_idx = i * features_per_link
            
            # Extract link features
            if self.with_density_obs:
                link_features = [
                    link.get_density(time_step), # shared density
                    link.inflow[time_step] if time_step < len(link.inflow) else 0.0, # inflow of outgoing link
                    link.reverse_link.outflow[time_step] if time_step < len(link.reverse_link.outflow) else 0.0, # outflow of incoming link
                ]
            else:
                link_features = [
                    link.inflow[time_step] if time_step < len(link.inflow) else 0.0, # inflow of outgoing link
                    link.reverse_link.outflow[time_step] if time_step < len(link.reverse_link.outflow) else 0.0, # outflow of incoming link
                ]
            
            obs[start_idx:start_idx + features_per_link] = link_features
        
        # Apply normalization if enabled
        if self.normalize:
            obs = self._normalize_gater_obs(obs)
        
        return obs
    
    def _normalize_separator_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize separator observation features."""
        # TODO: Apply proper normalization based on feature semantics
        # Placeholder normalization
        normalized = obs.copy()
        
        # Normalize density features (indices 0, 6)
        normalized[[0, 6]] /= self.density_norm
        
        # Normalize speed features (indices 1, 7)
        normalized[[1, 7]] /= self.speed_norm
        
        # Normalize travel time features (indices 2, 8)
        normalized[[2, 8]] /= self.time_norm
        
        # Normalize flow features (indices 4, 5, 10, 11)
        normalized[[4, 5, 10, 11]] /= self.flow_norm
        
        return normalized
    
    def _normalize_gater_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize gater observation features."""
        # TODO: Apply proper normalization for per-link features
        # Placeholder normalization
        normalized = obs.copy()
        
        features_per_link = 8
        max_outdegree = len(obs) // features_per_link
        
        for i in range(max_outdegree):
            start_idx = i * features_per_link
            
            # Normalize density (index 0)
            normalized[start_idx] /= self.density_norm
            
            # Normalize speed (index 1)
            normalized[start_idx + 1] /= self.speed_norm
            
            # Normalize travel time (index 2)
            normalized[start_idx + 2] /= self.time_norm
            
            # Normalize receiving flow (index 4)
            normalized[start_idx + 4] /= self.flow_norm
        
        return normalized


class ActionApplier:
    """
    Applies agent actions to network components.
    """
    
    def __init__(self, network, agent_discovery: AgentDiscovery, 
                 max_delta_sep_width: float = 0.1, max_delta_gate_width: float = 0.1, min_sep_width: float = 1.0):
        """
        Initialize action applier. for all kinds of algorithms, not just RL.
        
        Args:
            network: Network instance from LTM simulation
            agent_discovery: AgentDiscovery instance with agent mappings
            max_delta_sep_width: Maximum delta separator width within one time step
            max_delta_gate_width: Maximum delta gate width within one time step
            min_sep_width: Minimum separator width
        """
        self.network = network
        self.agent_discovery = agent_discovery
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
            agent_type = self.agent_discovery.get_agent_type(agent_id)
            
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
        if action_value < self.min_sep_width or action_value > forward_link.width - self.min_sep_width:
            return self.min_sep_width if action_value < self.min_sep_width else forward_link.width - self.min_sep_width
        if abs(action_value - forward_link.separator_width) > self.max_delta_sep_width:
            return forward_link.separator_width + self.max_delta_sep_width if action_value - forward_link.separator_width > self.max_delta_sep_width else forward_link.separator_width - self.max_delta_sep_width
        return action_value

    def clip_gater_action_value(self, action_value: float, link: Link):
        """
        Validate gater agent action value. if the action value is >=0 and <= max link width.
        If the change of the width is too large, clip the action value to the maximum or minimum value.
        """
        if action_value < 0 or action_value > link.width:
            return 0.0 if action_value < 0 else link.width
        if abs(action_value - link.back_gate_width) > self.max_delta_gate_width:
            return link.back_gate_width + self.max_delta_gate_width if action_value - link.back_gate_width > self.max_delta_gate_width else link.back_gate_width - self.max_delta_gate_width
        return action_value
    
    def _apply_separator_action(self, agent_id: str, action: np.ndarray):
        """
        Apply separator agent action to control lane allocation.
        
        Args:
            agent_id: Separator agent identifier
            action: Action array of shape (1,) is the processed action value: the actual width
        """
        # Get separator links
        forward_link, reverse_link = self.agent_discovery.get_separator_links(agent_id)
        # total_width = self.agent_discovery.get_separator_total_width(agent_id)
        
        # Convert action to width fraction, ensuring minimum width
        action_value = float(action[0]) # should be the actual width of the forward link
        action_value = self.clip_separator_action_value(action_value, forward_link)
        # max_frac = 1.0 - self.min_sep_frac
        # width_frac = self.min_sep_frac + action_value * (max_frac - self.min_sep_frac)
        
        # Set separator width (reverse width is automatically adjusted)
        # new_width = width_frac * total_width
        # forward_link.separator_width(action_value)
        forward_link.separator_width = action_value
    
    def _apply_gater_action(self, agent_id: str, action: np.ndarray):
        """
        Apply gater agent action to control gate widths.
        
        Args:
            agent_id: Gater agent identifier
            action: Action array of shape (num_outgoing_links,) with values in [0, 1]
        """
        # Get gater outgoing links
        out_links = self.agent_discovery.get_gater_outgoing_links(agent_id)
        
        # Apply actions to all outgoing links (no padding, direct mapping)
        for i, link in enumerate(out_links):
            # Action value is the actual width of the gate
            action_value = float(action[i])
            action_value = self.clip_gater_action_value(action_value, link)
            link.back_gate_width = action_value

