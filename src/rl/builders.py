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
        features.extend([
            forward_link.density[time_step] if time_step < len(forward_link.density) else 0.0 if self.with_density_obs else 0.0, # bidirectional density
            forward_link.inflow[time_step] if time_step < len(forward_link.inflow) else 0.0,
            forward_link.outflow[time_step] if time_step < len(forward_link.outflow) else 0.0,
        ])
        
        # Reverse direction features
        features.extend([
            reverse_link.density[time_step] if time_step < len(reverse_link.density) else 0.0 if self.with_density_obs else 0.0,
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
        node = self.agent_discovery.get_gater_node(agent_id)
        out_links = self.agent_discovery.get_gater_outgoing_links(agent_id)
        max_outdegree = self.agent_discovery.get_max_outdegree()
        
        # TODO: Extract per-outgoing-link features
        # Placeholder: 8 features per link, padded to max_outdegree
        features_per_link = 3 if self.with_density_obs else 2 # density, inflow of outgoing link, outflow of incoming link
        obs = np.zeros(max_outdegree * features_per_link, dtype=np.float32)
        
        for i, link in enumerate(out_links):
            start_idx = i * features_per_link
            
            # Extract link features
            link_features = [
                link.get_density(time_step) if self.with_density_obs else 0.0, # shared density
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
            elif agent_type == "gate":
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

