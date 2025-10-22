# -*- coding: utf-8 -*-
# @Time    : 26/01/2025 15:30
# @Author  : mmai
# @FileName: spaces
# @Software: PyCharm

"""
Action and observation space builders for PedNet multi-agent environment.

Defines Gymnasium spaces for separator and gater agents with proper bounds
and dimensionality.
"""

import numpy as np
from typing import Dict
from gymnasium import spaces
from .discovery import AgentDiscovery


class SpaceBuilder:
    """
    Builds action and observation spaces for discovered agents.
    """
    
    def __init__(self, agent_discovery: AgentDiscovery, with_density_obs: bool = False, min_sep_width: float = 1.0):
        """
        Initialize space builder.
        
        Args:
            agent_discovery: AgentDiscovery instance with network mappings
            with_density_obs: Whether density observations are included
            min_sep_width: Minimum width for each direction in bidirectional separators
        """
        self.agent_discovery = agent_discovery
        self.with_density_obs = with_density_obs
        self.min_sep_width = min_sep_width
        
        # Observation space dimensions based on actual features
        self.sep_obs_dim = 6 if with_density_obs else 4  # Forward + reverse link features
        self.gat_obs_dim_per_link = 3 if with_density_obs else 2  # Features per outgoing link
    
    def build_action_spaces(self) -> Dict[str, spaces.Space]:
        """Build action spaces for all agents with physical width constraints."""
        action_spaces = {}
        
        # Separator agents: control forward lane width in bidirectional corridor
        # Action value is actual width in meters: [min_sep_width, total_width - min_sep_width]
        for agent_id in self.agent_discovery.get_separator_agents():
            forward_link, reverse_link = self.agent_discovery.get_separator_links(agent_id)
            total_width = forward_link.width  # Total corridor width
            
            action_spaces[agent_id] = spaces.Box(
                low=self.min_sep_width,
                high=total_width - self.min_sep_width,
                shape=(1,),
                dtype=np.float32
            )
        
        # Gater agents: control gate widths for each outgoing link
        # Action value is actual gate width in meters: [0, link.width]
        for agent_id in self.agent_discovery.get_gater_agents():
            out_links = self.agent_discovery.get_gater_outgoing_links(agent_id)
            
            # Build bounds per outgoing link
            low_bounds = np.zeros(len(out_links), dtype=np.float32)
            high_bounds = np.array([link.width for link in out_links], dtype=np.float32)
            
            action_spaces[agent_id] = spaces.Box(
                low=low_bounds,
                high=high_bounds,
                shape=(len(out_links),),
                dtype=np.float32
            )
        
        return action_spaces
    
    def build_observation_spaces(self) -> Dict[str, spaces.Space]:
        """Build observation spaces for all agents."""
        observation_spaces = {}
        
        # Separator agents: fixed-size feature vector
        for agent_id in self.agent_discovery.get_separator_agents():
            observation_spaces[agent_id] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.sep_obs_dim,),
                dtype=np.float32
            )
        
        # Gater agents: per-outgoing-link features, padded to max_outdegree
        # Note: Currently using max_outdegree padding for observations (unlike actions)
        max_outdegree = self.agent_discovery.get_max_outdegree()
        for agent_id in self.agent_discovery.get_gater_agents():
            observation_spaces[agent_id] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(max_outdegree * self.gat_obs_dim_per_link,),
                dtype=np.float32
            )
        
        return observation_spaces
    
    def get_separator_obs_dim(self) -> int:
        """Return observation dimension for separator agents."""
        return self.sep_obs_dim
    
    def get_gater_obs_dim_per_link(self) -> int:
        """Return observation dimension per outgoing link for gater agents."""
        return self.gat_obs_dim_per_link
    
    def validate_separator_action(self, action: np.ndarray, agent_id: str) -> bool:
        """Validate separator agent action against physical bounds."""
        if not isinstance(action, np.ndarray) or action.shape != (1,):
            return False
        
        forward_link, _ = self.agent_discovery.get_separator_links(agent_id)
        total_width = forward_link.width
        
        return self.min_sep_width <= action[0] <= (total_width - self.min_sep_width)
    
    def validate_gater_action(self, action: np.ndarray, agent_id: str) -> bool:
        """Validate gater agent action against physical bounds."""
        out_links = self.agent_discovery.get_gater_outgoing_links(agent_id)
        
        if not isinstance(action, np.ndarray) or action.shape != (len(out_links),):
            return False
        
        # Check each action dimension against corresponding link width
        for i, link in enumerate(out_links):
            if not (0.0 <= action[i] <= link.width):
                return False
        
        return True

