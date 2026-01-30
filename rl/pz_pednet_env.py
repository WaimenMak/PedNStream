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

import torch


class PedNetParallelEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv for multi-agent pedestrian traffic control.
    
    Agents:
    - Separators (sep_u_v): control Separator.separator_width for bidirectional corridors
    - Gaters (gate_n): control Link.front_gate_width for outgoing links at nodes
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
        Compute rewards for all agents based on *changes* in throughput and delay,
        plus a density-based shaping term (for gaters).
    
        Per-step quantities for a gater agent a:
            - Throughput_t = Σ_out links link_flow_ℓ(t)
            - Delay_t      = Σ_in∪out links N_ℓ(t) * max(0, T_ℓ(t) - T_ℓ^free)
    
        We use deltas:
            ΔThroughput_t = Throughput_t - Throughput_{t-1}
            ΔDelay_t      = Delay_t      - Delay_{t-1}
    
        Reward:
            R_t = α * ΔThroughput_t - β * ΔDelay_t
                + density_reward - balance * penalty
        """
        rewards: Dict[str, float] = {}
    
        # Tunable weights
        alpha_throughput = 0.1  # change in throughput term (start at 0 if you only trust delay)
        beta_delay = 1.0        # change in delay term
        balance = 1.0            # weight for density-variance penalty
    
        for agent_id in self.possible_agents:
            agent_type = self.agent_manager.get_agent_type(agent_id)
    
            total_delay = 0.0
            total_throughput = 1
            density_reward = 0.0
            penalty = 0.0
    
            if agent_type == "gate":
                # Node and controlled outgoing links
                node = self.agent_manager.get_gater_node(agent_id)
                out_links = self.agent_manager.get_gater_outgoing_links(agent_id)
    
                # --- Outgoing links: delay + throughput ---
                for link in out_links:
                    # N_ell(t)
                    if self.sim_step < len(link.num_pedestrians):
                        N_ell = link.num_pedestrians[self.sim_step]
                    else:
                        N_ell = 0.0
    
                    # T_ell(t)
                    if self.sim_step < len(link.travel_time):
                        T_ell = link.travel_time[self.sim_step]
                    else:
                        T_ell = link.travel_time[0]
    
                    # T_ell^free
                    T_ell_free = link.length / link.free_flow_speed
    
                    # Delay in person-seconds
                    # delay = N_ell * max(0.0, T_ell - T_ell_free)
                    delay = T_ell
                    total_delay += delay
    
                    # Throughput proxy: link_flow(t)
                    if self.sim_step < len(link.outflow):
                        flow = link.link_flow[self.sim_step]
                    else:
                        flow = 0.0
                    total_throughput += flow
    
                # --- Incoming links: density shaping + extra delay ---
                all_densities = []
                for link in node.incoming_links:
                    # Skip virtual incoming links
                    if hasattr(link, "virtual_incoming_link") and link == link.virtual_incoming_link:
                        continue
    
                    density = link.get_density(self.sim_step)
    
                    # Your existing density_reward: if density increases above k_critical
                    if self.sim_step > 5:
                        last_density = link.get_density(self.sim_step - 1)
                        if density - last_density < 0 and density > link.k_critical:
                            density_reward += 10.0
    
                    normalized_density = density / (link.k_jam + 1e-6)
                    all_densities.append(normalized_density)
    
                    # Also add delay from incoming links
                    if self.sim_step < len(link.num_pedestrians):
                        N_ell = link.num_pedestrians[self.sim_step]
                    else:
                        N_ell = 0.0
    
                    if self.sim_step < len(link.travel_time):
                        T_ell = link.travel_time[self.sim_step]
                    else:
                        T_ell = link.travel_time[0]
    
                    T_ell_free = link.length / link.free_flow_speed
                    # delay = N_ell * max(0.0, T_ell - T_ell_free)
                    delay = T_ell
                    total_delay += delay
    
                    # throughput
                    if self.sim_step < len(link.outflow):
                        flow = link.link_flow[self.sim_step]
                    else:
                        flow = 0.0
                    total_throughput += flow
    
                # Optional variance penalty over densities around the node
                variance_penalty_weight = 10.0
                if len(all_densities) > 1:
                    avg_density = np.mean(all_densities)
                    # diff = np.mean(np.abs(np.array(all_densities) - avg_density))
                    diff = np.max(np.abs(np.array(all_densities) - avg_density))
                    penalty = variance_penalty_weight * diff
    
            # Keep original scale (person-seconds) for now
            delay_scaled = total_delay # convert to person-hours
    
            # --- Delta terms using stored previous totals ---
            prev_delay = self._prev_delay.get(agent_id, 0.0)
            prev_throughput = self._prev_throughput.get(agent_id, 0.0)
    
            if self.sim_step > 5 :
                delta_delay = delay_scaled - prev_delay
                delta_throughput = total_throughput - prev_throughput
            else:
                delta_delay = 0.0
                delta_throughput = 0.0
    
            # Action smoothness: penalty if action changes while density < critical
            action_diverse_reward = 0.0
            action_diverse_weight = 5.0  # Tunable weight
            all_below_critical = True
            for link in node.incoming_links:
    
                if link.get_density(self.sim_step) >= link.k_critical:
                    all_below_critical = False
                    break
    
            if not all_below_critical:
                try:
                    with torch.no_grad():
                        action_diff = abs(self.current_actions[agent_id] - self.last_actions[agent_id])
                        action_diverse_reward = action_diverse_weight * sum(action_diff)
                        action_diverse_reward = action_diverse_reward.item()
                except Exception:
                    action_diverse_reward = 0.0
    
            density_penalty = np.sum(all_densities)/len(all_densities)
            reward = (
                # alpha_throughput * delta_throughput
                # - beta_delay * delta_delay
                - beta_delay * delay_scaled
                # - balance * penalty
                # + density_reward
                # + action_diverse_reward
                # - density_penalty
            )
    
            # Update stored totals
            self._prev_delay[agent_id] = delay_scaled
            self._prev_throughput[agent_id] = total_throughput
    
            rewards[agent_id] = reward
    
        return rewards

    # def _compute_rewards(self) -> Dict[str, float]:
    #     """
    #     Compute rewards for all agents based on throughput maximization and delay minimization.

    #     Reward Function:
    #     R_t = α * Throughput - β * Delay
    #     where:
    #     - Throughput = Σ_{ℓ in L_controlled} outflow_ℓ(t) (people per time step)
    #     - Delay = Σ_{ℓ in L_controlled} delay_ℓ(t) / 3600 (person-hours)
    #     - delay_ℓ(t) = N_ℓ(t) * (T_ℓ(t) - T_ℓ^free) in seconds

    #     This encourages agents to maximize flow while minimizing delay.
    #     """
    #     rewards = {}
        
    #     # Weight parameters (tunable)
    #     alpha_throughput = 0  # Weight for throughput term
    #     beta_delay = 1        # Weight for delay term (in person-hours)
    #     balance = 0

    #     for agent_id in self.possible_agents:
    #         agent_type = self.agent_manager.get_agent_type(agent_id)
    #         total_delay = 0.0
    #         total_throughput = 0.0
    #         density_reward = 0.0
    #         if agent_type == 'gate':
    #             # Get node and its controlled outgoing links
    #             node = self.agent_manager.get_gater_node(agent_id)
    #             out_links = self.agent_manager.get_gater_outgoing_links(agent_id)

    #             # Compute delay and throughput for all outgoing links
    #             for link in out_links:
    #                 # Get current number of pedestrians
    #                 N_ell = link.num_pedestrians[self.sim_step] if self.sim_step < len(link.num_pedestrians) else 0.0
    #                 # N_ell = 1

    #                 # Get current travel time T_ℓ(t) (in seconds)
    #                 T_ell = link.travel_time[self.sim_step] if self.sim_step < len(link.travel_time) else link.travel_time[0]

    #                 # Get free-flow travel time T_ℓ^free (in seconds)
    #                 T_ell_free = link.length / link.free_flow_speed

    #                 # Compute delay: delay_ℓ(t) = N_ℓ(t) * (T_ℓ(t) - T_ell^free) in person-seconds
    #                 delay = N_ell * max(0, T_ell - T_ell_free)
    #                 total_delay += delay
                    
    #                 # Get throughput: outflow at current time step (people per time step)
    #                 flow = link.link_flow[self.sim_step] if self.sim_step < len(link.outflow) else 0.0
    #                 total_throughput += flow

    #             # Also consider incoming links (they affect the intersection)
    #             all_densities = []
    #             for link in node.incoming_links:
    #                 if hasattr(link, 'virtual_incoming_link') and link == link.virtual_incoming_link:
    #                     continue

    #                 density = link.get_density(self.sim_step)
    #                 if self.sim_step > 5:
    #                     last_density = link.get_density(self.sim_step - 1)
    #                     if density - last_density > 0 and density > link.k_critical:
    #                         density_reward += 10
                        
    #                 normalized_density = density / (link.k_jam + 1e-6)
    #                 all_densities.append(normalized_density)

    #                 # Get current number of pedestrians
    #                 N_ell = link.num_pedestrians[self.sim_step] if self.sim_step < len(link.num_pedestrians) else 0.0
    #                 # N_ell = 1

    #                 # Get current travel time (in seconds)
    #                 T_ell = link.travel_time[self.sim_step] if self.sim_step < len(link.travel_time) else link.travel_time[0]

    #                 # Get free-flow travel time (in seconds)
    #                 T_ell_free = link.length / link.free_flow_speed

    #                 # Compute delay (in person-seconds)
    #                 delay = N_ell * max(0, T_ell - T_ell_free)
    #                 total_delay += delay


    #             variance_penalty_weight = 1  # Tunable hyperparameter
    #             if len(all_densities) > 1:
    #                 avg_density = np.mean(all_densities)
    #                 # if avg_density > 0.5:
    #                 diff = np.mean(np.abs(np.array(all_densities) - avg_density))
    #                 # diff = np.max(np.abs(np.array(all_densities) - avg_density))
    #                 penalty = variance_penalty_weight * diff

            # elif agent_type == 'sep':
            #     # For separator, consider both forward and reverse links
            #     forward_link, reverse_link = self.agent_manager.get_separator_links(agent_id)
            #
            #     for link in [forward_link, reverse_link]:
            #         # Get current number of pedestrians
            #         N_ell = link.num_pedestrians[self.sim_step] if self.sim_step < len(link.num_pedestrians) else 0.0
            #
            #         # Get current travel time (in seconds)
            #         T_ell = link.travel_time[self.sim_step] if self.sim_step < len(link.travel_time) else link.travel_time[0]
            #
            #         # Get free-flow travel time (in seconds)
            #         T_ell_free = link.length / link.free_flow_speed
            #
            #         # Compute delay (in person-seconds)
            #         delay = N_ell * max(0, T_ell - T_ell_free)
            #         total_delay += delay
            #
            #         # Get throughput: outflow at current time step
            #         outflow = link.flow[self.sim_step] if self.sim_step < len(link.outflow) else 0.0
            #         total_throughput += outflow

            # Convert delay from person-seconds to person-hours (divide by 3600)
        #     delay_hours = total_delay
            
        #     # Combined reward: R_t = α * Throughput - β * Delay
        #     rewards[agent_id] = alpha_throughput * total_throughput - beta_delay * delay_hours - balance * penalty + density_reward

        # return rewards

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
