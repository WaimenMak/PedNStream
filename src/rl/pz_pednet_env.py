# -*- coding: utf-8 -*-
# @Time    : 26/01/2025 15:30
# @Author  : mmai
# @FileName: pz_pednet_env
# @Software: PyCharm

"""
PettingZoo ParallelEnv wrapper for PedNStream crowd control simulation.

Provides multi-agent RL environment with two controller types:
- Separators: control bidirectional lane allocation
- Gaters: control node-level gate widths
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pettingzoo import ParallelEnv
import gymnasium as gym
from gymnasium import spaces

from src.utils.env_loader import NetworkEnvGenerator
from .discovery import AgentDiscovery
from .spaces import SpaceBuilder
from .builders import ObservationBuilder, ActionApplier


class PedNetParallelEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv for multi-agent pedestrian traffic control.
    
    Agents:
    - Separators (sep:u_v): control Separator.separator_width for bidirectional corridors
    - Gaters (gat:n): control Link.front_gate_width for outgoing links at nodes
    """
    
    metadata = {"render_modes": ["human"], "name": "pednet_v0"}
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PedNet environment.
        
        Args:
            config: Configuration dictionary containing:
                - dataset: str, network dataset name (e.g., "delft", "melbourne")
                - simulation_steps: int, number of simulation time steps
                - controllers: dict, predefined controller configuration:
                    {
                        "separators": [(node1, node2), (node3, node4), ...],
                        "gaters": [node1, node2, node3, ...]
                    }
                - min_sep_frac: float, minimum separator width fraction [0, 0.5)
                - min_gate_frac: float, minimum gate width fraction [0, 1)
                - normalize_obs: bool, whether to normalize observations
                - early_stop_on_jam: bool, early termination on traffic jam
                - log_level: str, logging level for network simulation
                - seed: Optional[int], random seed
        """
        super().__init__()
        
        # Store configuration
        self.config = config
        self.dataset = config.get("dataset", "delft")
        self.simulation_steps = config.get("simulation_steps", 500)
        self.controller_config = config.get("controllers", {"separators": [], "gaters": []})
        self.min_sep_frac = config.get("min_sep_frac", 0.1)
        self.min_gate_frac = config.get("min_gate_frac", 0.1)
        self.normalize_obs = config.get("normalize_obs", True)
        self.early_stop_on_jam = config.get("early_stop_on_jam", False)
        self.log_level = config.get("log_level", "WARNING")
        
        # Initialize components (will be set in reset)
        self.env_generator = NetworkEnvGenerator()
        self.network = None
        self.agent_discovery = None
        self.space_builder = None
        self.obs_builder = None
        self.action_applier = None
        
        # Environment state
        self.current_step = 0
        self.agents_list = []
        self._agent_selector = None
        self._cumulative_rewards = {}
        
        # Spaces (will be built after agent discovery)
        self._action_spaces = {}
        self._observation_spaces = {}

    @property
    def agents(self) -> List[str]:
        """Return list of all agent IDs."""
        return self.agents_list.copy()

    def observation_space(self, agent: str) -> spaces.Space:
        """Return observation space for given agent."""
        if agent not in self._observation_spaces:
            raise ValueError(f"Agent {agent} not found in observation spaces")
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        """Return action space for given agent."""
        if agent not in self._action_spaces:
            raise ValueError(f"Agent {agent} not found in action spaces")
        return self._action_spaces[agent]

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        Reset the environment and return initial observations.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            Tuple of (observations, infos) dictionaries
        """
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Create fresh network environment
        self.network = self.env_generator.create_network(self.dataset)
        
        # TODO: Set network logging level to reduce training overhead
        # self.network.logger.setLevel(getattr(logging, self.log_level.upper()))
        
        # Create agents from predefined configuration
        self.agent_discovery = AgentDiscovery(self.network, self.controller_config)
        self.agents_list = self.agent_discovery.get_all_agent_ids()
        
        # Build action and observation spaces
        self.space_builder = SpaceBuilder(
            agent_discovery=self.agent_discovery,
            min_sep_frac=self.min_sep_frac,
            min_gate_frac=self.min_gate_frac
        )
        self._action_spaces = self.space_builder.build_action_spaces()
        self._observation_spaces = self.space_builder.build_observation_spaces()
        
        # Initialize observation builder and action applier
        self.obs_builder = ObservationBuilder(
            network=self.network,
            agent_discovery=self.agent_discovery,
            normalize=self.normalize_obs
        )
        self.action_applier = ActionApplier(
            network=self.network,
            agent_discovery=self.agent_discovery,
            min_sep_frac=self.min_sep_frac,
            min_gate_frac=self.min_gate_frac
        )
        
        # Reset environment state
        self.current_step = 1  # Network simulation starts at t=1
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents_list}
        
        # Build initial observations
        observations = self._get_observations()
        infos = self._get_infos()
        
        return observations, infos

    def step(self, actions: Dict[str, Any]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Execute one environment step with all agent actions.
        
        Args:
            actions: Dictionary mapping agent_id to action
            
        Returns:
            Tuple of (observations, rewards, terminations, truncations, infos)
        """
        # Validate actions
        for agent_id, action in actions.items():
            if agent_id not in self.agents_list:
                raise ValueError(f"Unknown agent: {agent_id}")
        
        # Apply all agent actions to the network
        self.action_applier.apply_all_actions(actions)
        
        # Advance the simulation by one step
        self.network.network_loading(self.current_step)
        
        # Build new observations
        observations = self._get_observations()
        
        # Compute rewards (placeholder for now)
        rewards = self._compute_rewards()
        
        # Check termination conditions
        terminations = self._check_terminations()
        truncations = self._check_truncations()
        
        # Build info dictionary
        infos = self._get_infos()
        
        # Update environment state
        self.current_step += 1
        for agent_id, reward in rewards.items():
            self._cumulative_rewards[agent_id] += reward
        
        return observations, rewards, terminations, truncations, infos

    def _get_observations(self) -> Dict[str, Any]:
        """Build observations for all agents."""
        observations = {}
        for agent_id in self.agents_list:
            observations[agent_id] = self.obs_builder.build_observation(agent_id, self.current_step)
        return observations

    def _compute_rewards(self) -> Dict[str, float]:
        """Compute rewards for all agents (placeholder)."""
        # TODO: Implement reward computation
        # Options: throughput, delay penalty, congestion avoidance, etc.
        rewards = {}
        for agent_id in self.agents_list:
            rewards[agent_id] = 0.0  # Placeholder
        return rewards

    def _check_terminations(self) -> Dict[str, bool]:
        """Check if any agents should terminate."""
        # Standard termination: reached simulation end
        terminated = self.current_step >= self.simulation_steps - 1
        
        # TODO: Optional early termination on severe congestion
        if self.early_stop_on_jam:
            # Check if any link is severely jammed
            pass
        
        return {agent_id: terminated for agent_id in self.agents_list}

    def _check_truncations(self) -> Dict[str, bool]:
        """Check if any agents should be truncated (time limits, etc.)."""
        # No truncation conditions for now
        return {agent_id: False for agent_id in self.agents_list}

    def _get_infos(self) -> Dict[str, Dict]:
        """Build info dictionaries for all agents."""
        infos = {}
        for agent_id in self.agents_list:
            info = {
                "step": self.current_step,
                "cumulative_reward": self._cumulative_rewards.get(agent_id, 0.0)
            }
            
            # Add action mask for gater agents
            if agent_id.startswith("gat:"):
                info["action_mask"] = self.agent_discovery.get_gater_action_mask(agent_id)
            
            infos[agent_id] = info
        
        return infos

    def render(self, mode="human"):
        """Render the environment (placeholder)."""
        # TODO: Optional visualization integration
        pass

    def close(self):
        """Clean up environment resources."""
        if self.network is not None:
            # TODO: Any cleanup needed for network simulation
            pass
