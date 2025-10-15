# -*- coding: utf-8 -*-
# @Time    : 13/10/2025 12:30
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
    
    def __init__(self, dataset: str):
        """
        Initialize the PedNet environment.
        
        Args:
            - dataset: str, network dataset name (e.g., "delft", "melbourne")
        """
        super().__init__()
        
        # Initialize components (will be set in reset)
        self.env_generator = NetworkEnvGenerator()
        
        self.network = self.env_generator.create_network(dataset)
        self.timestep = None
        self.sim_step = self.timestep + 1 # the simulation step starts from 1
        self.simulation_steps = self.network.params['simulation_steps']
        self.agents_list = []

        # initialize the observation builder and action applier
        self.obs_builder = ObservationBuilder(self.network, self.agent_discovery, self.normalize_obs)
        self.action_applier = ActionApplier(self.network, self.agent_discovery, self.min_sep_frac, self.min_gate_frac)

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
            self.network.demand_generator.seed = seed
        
        # Create fresh network environment
        self.network = self.env_generator.randomize_network(self.network, self.randomize_params)
        self.timestep = 0
        
        # TODO: Set network logging level to reduce training overhead
        # self.network.logger.setLevel(getattr(logging, self.log_level.upper()))
        
        
        # Reset environment state
        # self.current_step = 1  # Network simulation starts at t=1
        # self._cumulative_rewards = {agent: 0.0 for agent in self.agents_list}
        
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
        self.network.network_loading(self.sim_step)
        
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
        self.sim_step += 1
        for agent_id, reward in rewards.items():
            self._cumulative_rewards[agent_id] += reward
        
        return observations, rewards, terminations, truncations, infos

    def _get_observations(self) -> Dict[str, Any]:
        """Build observations for all agents."""
        observations = {}
        for agent_id in self.agents_list:
            observations[agent_id] = self.obs_builder.build_observation(agent_id, self.sim_step)
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
        terminated = self.sim_step >= self.simulation_steps - 1
        
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
                "step": self.sim_step,
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
