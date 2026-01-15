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
import functools

from src.utils.env_loader import NetworkEnvGenerator
from src.utils.visualizer import NetworkVisualizer, progress_callback
from .discovery import AgentManager
from .spaces import SpaceBuilder
from .builders import ObservationBuilder, ActionApplier

import matplotlib.pyplot as plt
import matplotlib
from handlers.output_handler import OutputHandler
from matplotlib.animation import PillowWriter
import os


class PedNetParallelEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv for multi-agent pedestrian traffic control.
    
    Agents:
    - Separators (sep_u_v): control Separator.separator_width for bidirectional corridors
    - Gaters (gate_n): control Link.front_gate_width for outgoing links at nodes
    """
    
    metadata = {"render_modes": ["human", "animate"], "name": "pednet_v0"}
    
    def __init__(self, dataset: str, normalize_obs: bool = False, obs_mode: str = "option1",
                 render_mode: Optional[str] = None, verbose: bool = False):
        """
        Initialize the PedNet environment.
        
        Args:
            dataset: str, network dataset name (e.g., "delft", "melbourne")
            normalize_obs: Whether to normalize observations
            obs_mode: Observation mode - one of: "option1", "option2", "option3", "option4"
            render_mode: Rendering mode ("human", "animate", or None)
            verbose: Whether to enable logging output. Default False for RL training.
        """
        super().__init__()
        
        # Store render mode (required by PettingZoo API)
        self.render_mode = render_mode
        self.verbose = verbose
        
        # Initialize components (will be set in reset)
        self.env_generator = NetworkEnvGenerator()
        self.dataset = dataset
        
        self.network = self.env_generator.create_network(dataset, verbose=verbose)
        # self.timestep = 0
        self.sim_step = 1  # Network simulation starts at t=1
        self.simulation_steps = self.network.params['simulation_steps']
        self._max_delta_sep_width = 0.25 * self.network.params['unit_time'] # 0.25 meters per sec
        self._max_delta_gate_width = 0.25 * self.network.params['unit_time'] # 0.25 meters per sec
        self._min_sep_width = 1.5  # Minimum width for each direction in bidirectional corridors (meters)
        
        # Discover agents from network configuration
        self.agent_manager = AgentManager(self.network)
        self.possible_agents = self.agent_manager.get_all_agent_ids()
        
        # Build action and observation spaces
        self.normalize_obs = normalize_obs
        self.obs_mode = obs_mode

        # Initialize observation builder and action applier
        self.obs_builder = ObservationBuilder(self.network, self.agent_manager, self.normalize_obs, self.obs_mode)
        self.action_applier = ActionApplier(self.network, self.agent_manager, self._max_delta_sep_width, self._max_delta_gate_width, self._min_sep_width)

        self.space_builder = SpaceBuilder(self.agent_manager, self.obs_mode, self._min_sep_width)
        self._action_spaces = self.space_builder.build_action_spaces()
        self._observation_spaces = self.space_builder.build_observation_spaces(self.obs_builder.features_per_link)
        
        
        # Initialize cumulative rewards
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}

        # Initialize visualizer
        self.visualizer = None

    @property
    def agents(self) -> List[str]:
        """Return list of all agent IDs."""
        return self.possible_agents.copy()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        """Return observation space for given agent."""
        if agent not in self._observation_spaces:
            raise ValueError(f"Agent {agent} not found in observation spaces")
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        """Return action space for given agent."""
        if agent not in self._action_spaces:
            raise ValueError(f"Agent {agent} not found in action spaces")
        return self._action_spaces[agent]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """
        Reset the environment and return initial observations.
        
        Args:
            seed: Random seed for reproducibility
            options: Optional dictionary with reset options. Supported keys:
                - 'randomize' (bool): Whether to randomize the network at reset (default: False)
        Returns:
            Tuple of (observations, infos) dictionaries
        """
        # Extract options
        randomize = options.get('randomize', False) if options else False
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            if hasattr(self.network, 'demand_generator'):
                self.network.demand_generator.seed = seed
        
        # Determine reset mode
        # Always re-create network to clear state
        if randomize:
            self.network = self.env_generator.randomize_network(self.dataset, seed, verbose=self.verbose)
        else:
            # Deterministic reset using default configuration
            self.network = self.env_generator.create_network(self.dataset, verbose=self.verbose)
            
        # Re-initialize components with the new network instance
        self.agent_manager = AgentManager(self.network)
        
        # Verify agents haven't changed (topology should be constant)
        # new_agents = set(self.agent_manager.get_all_agent_ids())
        # if new_agents != set(self.possible_agents):
        #     self.possible_agents = list(new_agents)
            
        # Update builders/appliers with new references
        self.obs_builder = ObservationBuilder(self.network, self.agent_manager, self.normalize_obs, self.obs_mode)
        self.action_applier = ActionApplier(self.network, self.agent_manager, self._max_delta_sep_width, self._max_delta_gate_width, self._min_sep_width)
        
        # Reset environment state
        # self.timestep = 0
        self.sim_step = 1  # Network simulation starts at t=1
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
        
        # Build initial observations
        observations = self._get_observations()
        infos = self._get_infos()
        
        return observations, infos

    def step(self, actions: Dict[str, Any]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Execute one environment step with all agent actions.
        
        Args:
            actions: Dictionary mapping agent_id to action, the value is the width of the separator or gate
            
        Returns:
            Tuple of (observations, rewards, terminations, truncations, infos)
        """
        # Validate actions
        for agent_id, action in actions.items():
            if agent_id not in self.possible_agents:
                raise ValueError(f"Unknown agent: {agent_id}")
        
        # Apply all agent actions to the network
        if len(actions) > 0:    
            self.action_applier.apply_all_actions(actions)
        else:
            print("No actions provided, skipping action application.")
        
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
        for agent_id in self.possible_agents:
            observations[agent_id] = self.obs_builder.build_observation(agent_id, self.sim_step)
        return observations

    def _compute_rewards(self) -> Dict[str, float]:
        """
        Compute rewards for all agents based on intersection density minimization.
        
        Reward Function:
        R_t = - sum_{l in L_in U L_out} (density_l / k_jam_l)^2
        
        This encourages the agent to balance density across the intersection,
        preventing both upstream queuing and downstream congestion.
        """
        rewards = {}
        threshold_density = 4
        for agent_id in self.possible_agents:
            agent_type = self.agent_manager.get_agent_type(agent_id)
            penalty = 0.0
            
            if agent_type == 'gate':
                # Get node and its connected links
                node = self.agent_manager.get_gater_node(agent_id)
                
                # Collect all real (non-virtual) incoming links
                for link in node.incoming_links:
                    if hasattr(link, 'virtual_incoming_link') and link == link.virtual_incoming_link:
                        continue
                    # For regular bidirectional Links, get_density() returns combined density
                    # For Separators or unidirectional links, use directional density
                    from src.LTM.link import Separator
                    if isinstance(link, Separator):
                        # Separator: use directional density (independent lanes)
                        norm_reverse_link_density = link.reverse_link.density[self.sim_step] / (link.reverse_link.k_jam + 1e-6)
                        norm_forward_link_density = link.density[self.sim_step] / (link.k_jam + 1e-6)
                        normalized_density = (norm_reverse_link_density + norm_forward_link_density) / 2
                    else:
                        # Regular Link: use get_density() to get combined bidirectional density once
                        density = link.get_density(self.sim_step)
                        # high penalty for high density
                        if density > threshold_density:
                            penalty += 100 * (density - threshold_density)
                        normalized_density = density / (link.k_jam + 1e-6)
                        # normalized_density = link.get_density(self.sim_step)

                    penalty += normalized_density
                # average the penalty of all incoming links
                penalty /= len(node.incoming_links)
                
                # Add variance penalty to discourage blocking behavior
                # High variance = unbalanced load (e.g., one link blocked, others empty)
                # Low variance = balanced load (desired)
                penalty = 0
                variance_penalty_weight = 10  # Tunable hyperparameter
                if len(all_densities) > 1:
                    avg_density = np.mean(all_densities)
                    # if avg_density > 0.5:
                    diff = np.mean(np.abs(np.array(all_densities) - avg_density))
                    # diff = np.max(np.abs(np.array(all_densities) - avg_density))
                    penalty += variance_penalty_weight * diff

                    
            
            elif agent_type == 'sep':
                # For separator, consider both directions of the bidirectional corridor
                forward_link, reverse_link = self.agent_manager.get_separator_links(agent_id)
                
                for link in [forward_link, reverse_link]:
                    normalized_density = link.density[self.sim_step] / (link.k_jam + 1e-6)
                    penalty += normalized_density
                # average the penalty of all links
                penalty /= 2
            
            # Reward is negative penalty (minimize congestion)
            rewards[agent_id] = -penalty
        
        return rewards

    def _check_terminations(self) -> Dict[str, bool]:
        """Check if any agents should terminate."""
        # Standard termination: reached simulation end
        terminated = self.sim_step >= self.simulation_steps
        
        # TODO: Optional early termination on severe congestion
        # if self.early_stop_on_jam:
        #     # Check if any link is severely jammed
        #     pass
        
        return {agent_id: terminated for agent_id in self.possible_agents}

    def _check_truncations(self) -> Dict[str, bool]:
        """Check if any agents should be truncated (time limits, etc.)."""
        # No truncation conditions for now
        return {agent_id: False for agent_id in self.possible_agents}

    def _get_infos(self) -> Dict[str, Dict]:
        """Build info dictionaries for all agents."""
        infos = {}
        for agent_id in self.possible_agents:
            info = {
                "step": self.sim_step,
                "cumulative_reward": self._cumulative_rewards.get(agent_id, 0.0)
            }
            
            infos[agent_id] = info
        
        return infos

    def render(self, simulation_dir: str = None, variable = 'density', vis_actions: bool = False, save_dir: str = None):
        """Render the environment based on mode."""
        # Use self.render_mode if mode not specified (PettingZoo standard)
        if self.render_mode is None:
            return  # No rendering if render_mode is None
        
        if simulation_dir is not None:  
            self.visualizer = NetworkVisualizer(simulation_dir=simulation_dir, pos=self.network.pos)
            # When loading from saved data, let visualizer determine end_time from saved params
            end_time = None
        else:
            self.visualizer = NetworkVisualizer(network=self.network, pos=self.network.pos)
            # When using live network, use current simulation step
            end_time = self.sim_step
            
        if self.render_mode == "human":
            # Static blocking display of current state
            self.visualizer.visualize_network_state(
                time_step=end_time if end_time else self.sim_step,
                edge_property=variable,  # Customize as needed (e.g., 'flow', 'speed')
                with_colorbar=True,
                set_title=True,
                figsize=(10, 8)
            )
        elif self.render_mode == "animate":
            # Full animation from start to end
            matplotlib.use('macosx')
            ani = self.visualizer.animate_network(
                start_time=0,
                end_time=end_time,  # None = use saved simulation length, else current step
                interval=100,  # Adjust speed as needed
                edge_property=variable,  # Customize as needed
                tag=False,  # Optional labels
                vis_actions=vis_actions
            )
            plt.show()  # Blocks until animation is closed
            if save_dir is not None:
                writer = PillowWriter(fps=10, metadata=dict(artist='Me'))
                ani.save(os.path.join(save_dir, f"{self.dataset}_{self.sim_step}.gif"),
                         writer=writer,
                         progress_callback=progress_callback)
        else:
            raise ValueError(f"Unsupported render mode: {self.render_mode}")

    def save(self, simulation_dir: str):
        """Save the current network state."""
        output_handler = OutputHandler(base_dir="../outputs", simulation_dir=simulation_dir)
        output_handler.save_network_state(self.network)

    def close(self):
        """Clean up environment resources."""
        if self.network is not None:
            # TODO: Any cleanup needed for network simulation
            pass
