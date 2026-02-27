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
import random
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from pettingzoo import ParallelEnv

from gymnasium import spaces
import functools

# from src.utils.env_loader import NetworkEnvGenerator
# from src.utils.visualizer import NetworkVisualizer, progress_callback
from pednstream.utils.env_loader import NetworkEnvGenerator
from pednstream.utils.visualizer import NetworkVisualizer, progress_callback
from .discovery import AgentManager
from .spaces import SpaceBuilder
from .builders import ObservationBuilder, ActionApplier

import matplotlib.pyplot as plt
import matplotlib
# from handlers.output_handler import OutputHandler
from pednstream.utils import OutputHandler
from matplotlib.animation import PillowWriter
import os

import torch


class PedNetParallelEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv for multi-agent pedestrian traffic control.
    
    Agents:
    - Separators (sep_u_v): control Separator.separator_width for bidirectional corridors
    - Gaters (gate_n): control Link.front_gate_width and Link.back_gate_width for outgoing links at nodes
    """
    
    metadata = {"render_modes": ["human", "animate"], "name": "pednet_v0"}
    
    def __init__(self, dataset: str, normalize_obs: bool = False, obs_mode: str = "option1",
                 render_mode: Optional[str] = None, verbose: bool = False, action_gap: int = 1,
                 seed: Optional[int] = None):
        """
        Initialize the PedNet environment.
        
        Args:
            dataset: str, network dataset name (e.g., "delft", "melbourne")
            normalize_obs: Whether to normalize observations
            obs_mode: Observation mode - one of: "option1", "option2", "option3", "option4"
            render_mode: Rendering mode ("human", "animate", or None)
            verbose: Whether to enable logging output. Default False for RL training.
            action_gap: Number of steps between applying actions. Default 1.
            seed: Random seed for reproducibility. Set once at construction time.
        """
        super().__init__()
        
        # Store render mode (required by PettingZoo API)
        self.render_mode = render_mode
        self.verbose = verbose
        
        # Set random seed for reproducibility (set once at construction)
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
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
        # Track previous totals for delta-based rewards (per agent)
        self._prev_delay = {agent: 0.0 for agent in self.possible_agents}
        self._prev_throughput = {agent: 0.0 for agent in self.possible_agents}
        # every action_gap steps, apply the actions
        self._action_gap = action_gap
        self.last_actions = None
        self.current_actions = None

        # Initialize visualizer
        self.visualizer = None
    
    def seed(self, seed: int) -> None:
        """Set the random seed for reproducibility."""
        self._seed = seed
        np.random.seed(seed)
        random.seed(seed)

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
            seed: Ignored. For API compatibility only. Set seed in __init__ instead.
            options: Optional dictionary with reset options. Supported keys:
                - 'randomize' (bool): Whether to randomize the network at reset (default: False)
        Returns:
            Tuple of (observations, infos) dictionaries
        """
        # Extract options
        randomize = options.get('randomize', False) if options else False
        
        # Note: seed parameter is ignored. Seed should be set at construction time via __init__.
        # This maintains API compatibility with PettingZoo but ensures reproducibility
        # is controlled at environment creation, not at each reset.
        
        # Determine reset mode
        # Always re-create network to clear state
        if randomize:
            # Use stored seed for randomization (set at construction time)
            self.network = self.env_generator.randomize_network(self.dataset, seed=None, verbose=self.verbose)
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
        self._prev_delay = {agent: 0.0 for agent in self.possible_agents}
        self._prev_throughput = {agent: 0.0 for agent in self.possible_agents}
        
        # Late-start: fast-forward the simulation without agent control
        # This creates diverse starting conditions (system already populated with pedestrians)
        late_start_prob = options.get('late_start_prob', self.late_start_prob) if options else self.late_start_prob
        late_start_max_frac = options.get('late_start_max_frac', self.late_start_max_frac) if options else self.late_start_max_frac
        late_start_step = 1  # default: no late start
        
        if late_start_prob > 0.0 and np.random.random() < late_start_prob:
            max_skip = max(1, int(self.simulation_steps * late_start_max_frac))
            skip_steps = np.random.randint(1, max_skip + 1)
            # Run simulation without agent actions for skip_steps
            for t in range(1, skip_steps + 1):
                self.network.network_loading(t)
            self.sim_step = skip_steps + 1
            late_start_step = self.sim_step
        
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
        # record the actions for current step
        self.current_actions = actions
        if self.last_actions is None:
            self.last_actions = actions

        # Validate actions
        for agent_id, action in actions.items():
            if agent_id not in self.possible_agents:
                raise ValueError(f"Unknown agent: {agent_id}")
        
        # Apply all agent actions to the network
        if len(actions) > 0:    
            self.action_applier.apply_all_actions(actions)
        else:
            if self.sim_step == 1:
                print("No actions provided, skipping action application.")

        # Initialize cumulative rewards for this action gap
        cumulative_rewards = {agent_id: 0.0 for agent_id in self.possible_agents}
        
        for _ in range(self._action_gap): # every action_gap steps, apply the actions
            # Advance the simulation by one step
            self.network.network_loading(self.sim_step)
            
            # Build new observations
            observations = self._get_observations()
            
            # Compute rewards (placeholder for now)
            step_rewards = self._compute_rewards()
            
            # Accumulate rewards for each agent
            for agent_id, reward in step_rewards.items():
                cumulative_rewards[agent_id] += reward
            
            # Check termination conditions
            terminations = self._check_terminations()
            truncations = self._check_truncations()
            
            # Build info dictionary
            infos = self._get_infos()
            
            # Update environment state
            self.sim_step += 1
        
        # Use the cumulative rewards for this action gap
        rewards = cumulative_rewards
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
        Compute rewards for all agents based on throughput maximization and delay minimization.
        """
        rewards = {}
        for agent_id in self.possible_agents:
            agent_type = self.agent_manager.get_agent_type(agent_id)
            if agent_type == 'gate':
                node = self.agent_manager.get_gater_node(agent_id)
                out_links = self.agent_manager.get_gater_outgoing_links(agent_id)
                agent_rewards = 0.0
                all_densities = []
                tt_term = 0
                flow_term = 0
                for link in out_links:
                    # link_flow = 0.0
                    # link_travel_time = 0.0
                    link_excess_density_penalty = 0.0
                    density = link.get_density(self.sim_step)
                    all_densities.append(density/link.k_jam)
                    T_ell = link.travel_time[self.sim_step] if self.sim_step < len(link.travel_time) else link.travel_time[0]
                    T_ell_reverse = link.reverse_link.travel_time[self.sim_step] if self.sim_step < len(link.reverse_link.travel_time) else link.reverse_link.travel_time[0]
                    link_flow_forward = link.link_flow[self.sim_step] if self.sim_step < len(link.outflow) else 0.0
                    link_flow_reverse = link.reverse_link.link_flow[self.sim_step] if self.sim_step < len(link.reverse_link.outflow) else 0.0
                    link_flow = link_flow_forward + link_flow_reverse
                    T_free = link.length/link.free_flow_speed
                    # number of pedestrians
                    # num_peds = link.num_pedestrians[self.sim_step] if self.sim_step < len(link.num_pedestrians) else 0.0
                    # num_peds += link.reverse_link.num_pedestrians[self.sim_step] if self.sim_step < len(link.reverse_link.num_pedestrians) else 0.0
                    # normed_density = density/link.k_jam
                    link_travel_time = T_ell + T_ell_reverse
                    
                    # normalize the travel time by the free flow travel time
                    # T_max = 1000
                    norm_link_travel_time = np.clip(np.log(link_travel_time / 2 / T_free), 0, 2)
                    norm_link_flow = np.clip((link_flow / 2) / (link.free_flow_speed * link.k_critical), 0, 1)
                    # print(norm_link_travel_time, norm_link_flow)
                    # link_rewards -= norm_link_travel_time
                    # link_rewards += norm_link_flow
                    tt_term -= norm_link_travel_time
                    flow_term += norm_link_flow

                # Fairness
                diff_term = 0.0
                if len(all_densities) > 1 and np.max(all_densities) > 0.6:
                    avg_density = np.mean(all_densities)
                    diff_term = -np.mean(np.abs(np.array(all_densities) - avg_density))
                    # penalty for norm density larger than 0.6
                    # diff_term = -np.sum(np.maximum(np.array(all_densities) - 0.6, 0))
                    # diff = np.var(all_densities)
                    # penalty = variance_penalty_weight * diff
                    # link_rewards -= penalty

                w1 = 2.0  # throughput
                w2 = 1.0  # delay
                w3 = 0.01  # fairness
                agent_rewards = w1*flow_term + w2*tt_term + w3*diff_term
                # print(f"Agent {agent_id} reward components: flow={flow_term:.3f}, tt={tt_term:.3f}, diff={diff_term:.3f}, total={agent_rewards:.3f}")
                rewards[agent_id] = agent_rewards
        return rewards

    def _check_terminations(self) -> Dict[str, bool]:
        """
        Check if any agents should terminate.
        
        Termination conditions:
        1. Standard: Reached simulation end
        2. Jam condition: All links controlled by an agent reach jam density
        """
        # Standard termination: reached simulation end
        terminated = self.sim_step >= self.simulation_steps
        
        # Early termination on severe jam: Check if any agent has all its links jammed
        # if not terminated and self.sim_step > 0:
        #     for agent_id in self.possible_agents:
        #         agent_type = self.agent_manager.get_agent_type(agent_id)
                
        #         if agent_type == "sep":
        #             # Separator agent: check both forward and reverse links
        #             forward_link, reverse_link = self.agent_manager.get_separator_links(agent_id)
        #             links_to_check = [forward_link, reverse_link]
        #         elif agent_type == "gate":
        #             # Gater agent: check all outgoing links
        #             links_to_check = self.agent_manager.get_gater_outgoing_links(agent_id)
        #         else:
        #             continue
                
        #         # Check if ALL links for this agent are at jam density
        #         all_jammed = True
        #         for link in links_to_check:
        #             current_density = link.get_density(self.sim_step)
        #             # Consider jammed if density >= 95% of jam density
        #             if current_density < 0.99 * link.k_jam:
        #                 all_jammed = False
        #                 break
                
        #         if all_jammed:
        #             # This agent's links are all jammed - terminate episode
        #             terminated = True
        #             print(f"Agent {agent_id} is terminated because all its links are jammed.")
        #             break
        
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
