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
    
    def __init__(self, agent_discovery: AgentDiscovery):
        """
        Initialize space builder.
        
        Args:
            agent_discovery: AgentDiscovery instance with network mappings
        """
        self.agent_discovery = agent_discovery
        
        # Observation space dimensions (placeholders - will be refined)
        self.sep_obs_dim = 12  # TODO: Define based on actual features
        self.gat_obs_dim = 8   # TODO: Define based on actual features per outgoing link
    
    def build_action_spaces(self) -> Dict[str, spaces.Space]:
        """Build action spaces for all agents."""
        action_spaces = {}
        
        # Separator agents: continuous scalar in [0, 1]
        for agent_id in self.agent_discovery.get_separator_agents():
            action_spaces[agent_id] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            )
        
        # Gater agents: continuous vector sized to actual number of outgoing links
        for agent_id in self.agent_discovery.get_gater_agents():
            num_outgoing = len(self.agent_discovery.get_gater_outgoing_links(agent_id))
            action_spaces[agent_id] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(num_outgoing,),
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
        max_outdegree = self.agent_discovery.get_max_outdegree()
        for agent_id in self.agent_discovery.get_gater_agents():
            observation_spaces[agent_id] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(max_outdegree * self.gat_obs_dim,),
                dtype=np.float32
            )
        
        return observation_spaces
    
    def get_separator_obs_dim(self) -> int:
        """Return observation dimension for separator agents."""
        return self.sep_obs_dim
    
    def get_gater_obs_dim_per_link(self) -> int:
        """Return observation dimension per outgoing link for gater agents."""
        return self.gat_obs_dim
    
    def validate_separator_action(self, action: np.ndarray) -> bool:
        """Validate separator agent action."""
        return (isinstance(action, np.ndarray) and 
                action.shape == (1,) and 
                0.0 <= action[0] <= 1.0)
    
    def validate_gater_action(self, action: np.ndarray, agent_id: str) -> bool:
        """Validate gater agent action."""
        num_outgoing = len(self.agent_discovery.get_gater_outgoing_links(agent_id))
        return (isinstance(action, np.ndarray) and 
                action.shape == (num_outgoing,) and 
                np.all((0.0 <= action) & (action <= 1.0)))

