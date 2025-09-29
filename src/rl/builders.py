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


class ObservationBuilder:
    """
    Builds observations for separator and gater agents from network state.
    """
    
    def __init__(self, network, agent_discovery: AgentDiscovery, normalize: bool = True):
        """
        Initialize observation builder.
        
        Args:
            network: Network instance from LTM simulation
            agent_discovery: AgentDiscovery instance with agent mappings
            normalize: Whether to normalize observations
        """
        self.network = network
        self.agent_discovery = agent_discovery
        self.normalize = normalize
        
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
        elif agent_type == "gat":
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
        features.extend([
            forward_link.density[time_step] if time_step < len(forward_link.density) else 0.0,
            forward_link.speed[time_step] if time_step < len(forward_link.speed) else 0.0,
            forward_link.avg_travel_time[time_step] if time_step < len(forward_link.avg_travel_time) else 0.0,
            forward_link.num_pedestrians[time_step] if time_step < len(forward_link.num_pedestrians) else 0.0,
            forward_link.sending_flow[time_step-1] if time_step > 0 and time_step-1 < len(forward_link.sending_flow) else 0.0,
            forward_link.receiving_flow[time_step-1] if time_step > 0 and time_step-1 < len(forward_link.receiving_flow) else 0.0
        ])
        
        # Reverse direction features
        features.extend([
            reverse_link.density[time_step] if time_step < len(reverse_link.density) else 0.0,
            reverse_link.speed[time_step] if time_step < len(reverse_link.speed) else 0.0,
            reverse_link.avg_travel_time[time_step] if time_step < len(reverse_link.avg_travel_time) else 0.0,
            reverse_link.num_pedestrians[time_step] if time_step < len(reverse_link.num_pedestrians) else 0.0,
            reverse_link.sending_flow[time_step-1] if time_step > 0 and time_step-1 < len(reverse_link.sending_flow) else 0.0,
            reverse_link.receiving_flow[time_step-1] if time_step > 0 and time_step-1 < len(reverse_link.receiving_flow) else 0.0
        ])
        
        obs = np.array(features, dtype=np.float32)
        
        # Apply normalization if enabled
        if self.normalize:
            obs = self._normalize_separator_obs(obs)
        
        return obs
    
    def _build_gater_observation(self, agent_id: str, time_step: int) -> np.ndarray:
        """Build observation for gater agent."""
        node = self.agent_discovery.get_gater_node(agent_id)
        out_links = self.agent_discovery.get_gater_outgoing_links(agent_id)
        max_outdegree = self.agent_discovery.get_max_outdegree()
        
        # TODO: Extract per-outgoing-link features
        # Placeholder: 8 features per link, padded to max_outdegree
        features_per_link = 8
        obs = np.zeros(max_outdegree * features_per_link, dtype=np.float32)
        
        for i, link in enumerate(out_links):
            start_idx = i * features_per_link
            
            # Extract link features
            link_features = [
                link.density[time_step] if time_step < len(link.density) else 0.0,
                link.speed[time_step] if time_step < len(link.speed) else 0.0,
                link.avg_travel_time[time_step] if time_step < len(link.avg_travel_time) else 0.0,
                link.num_pedestrians[time_step] if time_step < len(link.num_pedestrians) else 0.0,
                link.receiving_flow[time_step-1] if time_step > 0 and time_step-1 < len(link.receiving_flow) else 0.0,
                link.front_gate_width / link.width if link.width > 0 else 1.0,  # Current gate width ratio
                link.capacity,  # Link capacity
                0.0  # Placeholder for additional feature
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
                 min_sep_frac: float = 0.1, min_gate_frac: float = 0.1):
        """
        Initialize action applier.
        
        Args:
            network: Network instance from LTM simulation
            agent_discovery: AgentDiscovery instance with agent mappings
            min_sep_frac: Minimum separator width fraction
            min_gate_frac: Minimum gate width fraction
        """
        self.network = network
        self.agent_discovery = agent_discovery
        self.min_sep_frac = min_sep_frac
        self.min_gate_frac = min_gate_frac
    
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
            elif agent_type == "gat":
                self._apply_gater_action(agent_id, action)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
    
    def _apply_separator_action(self, agent_id: str, action: np.ndarray):
        """
        Apply separator agent action to control lane allocation.
        
        Args:
            agent_id: Separator agent identifier
            action: Action array of shape (1,) with value in [0, 1]
        """
        # Get separator links
        forward_link, reverse_link = self.agent_discovery.get_separator_links(agent_id)
        total_width = self.agent_discovery.get_separator_total_width(agent_id)
        
        # Convert action to width fraction, ensuring minimum width
        action_value = float(action[0])
        max_frac = 1.0 - self.min_sep_frac
        width_frac = self.min_sep_frac + action_value * (max_frac - self.min_sep_frac)
        
        # Set separator width (reverse width is automatically adjusted)
        new_width = width_frac * total_width
        forward_link.separator_width = new_width
    
    def _apply_gater_action(self, agent_id: str, action: np.ndarray):
        """
        Apply gater agent action to control gate widths.
        
        Args:
            agent_id: Gater agent identifier
            action: Action array of shape (max_outdegree,) with values in [0, 1]
        """
        # Get gater outgoing links
        out_links = self.agent_discovery.get_gater_outgoing_links(agent_id)
        action_mask = self.agent_discovery.get_gater_action_mask(agent_id)
        
        # Apply actions only to valid outgoing links
        for i, link in enumerate(out_links):
            if action_mask[i] > 0:  # Valid action slot
                # Convert action to gate width fraction
                action_value = float(action[i])
                max_frac = 1.0 - self.min_gate_frac
                gate_frac = self.min_gate_frac + action_value * (max_frac - self.min_gate_frac)
                
                # Set front gate width
                new_gate_width = gate_frac * link.width
                link.front_gate_width = new_gate_width

