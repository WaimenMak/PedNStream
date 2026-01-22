# -*- coding: utf-8 -*-
# @Time    : 23/10/2025 13:40
# @Author  : mmai
# @FileName: rule_based
# @Software: PyCharm

import numpy as np
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    @abstractmethod
    def take_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Takes an observation and returns an action.
        """
        pass


class RuleBasedGaterAgent(BaseAgent):
    """
    A rule-based agent for controlling gaters based on pressure differential.

    The gate will tend to close if the density of it's link is higher than a threshold. And the inflow
    of the gate is larger than the outflow of the link. Otherwise, the gate will tend to open.
    """
    def __init__(self, outgoing_links: list, obs_mode: str, threshold_density: float = 0.8):
        if obs_mode != "option2":
            raise ValueError("RuleBasedGaterAgent requires density information ('obs_mode' must be 'option2') with density observation.")
        self.outgoing_links = outgoing_links
        self.threshold_density = threshold_density
        self.features_per_link = 4  # density, inflow, reverse_outflow, current_width

    def take_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Calculates actions based on basic rules.

        Args:
            obs (np.ndarray): The observation for this agent. It's a flattened
                              array of features for each outgoing link, padded to
                              the max out-degree in the network.

        Returns:
            np.ndarray: An array of target gate widths for each outgoing link.
        """
        # First, calculate average downstream density across all outgoing links
        # downstream_densities = []
        # for i in range(len(self.outgoing_links)):
        #     start_idx = i * self.features_per_link
        #     link_obs = obs[start_idx: start_idx + self.features_per_link]
        #     # The observation vector is structured as:
        #     # [inflow, reverse_outflow, density, current_width]
        #     density = link_obs[2]  # density
        #     downstream_densities.append(density)
        #
        # avg_downstream_density = np.mean(downstream_densities) if downstream_densities else 0.0
        #
        # # If average downstream density is not higher than threshold, open all gates to max width
        # if avg_downstream_density <= 2:
        #     actions = [link.width for link in self.outgoing_links]
        #     return np.array(actions, dtype=np.float32)
        
        # Otherwise, apply per-link logic
        actions = []
        for i in range(len(self.outgoing_links)):
            start_idx = i * self.features_per_link
            link_obs = obs[start_idx: start_idx + self.features_per_link]

            # The observation vector is structured as:
            # [inflow, reverse_outflow, density, current_width]
            density = link_obs[2]        # density
            outflow = link_obs[1]          # reverse_link.outflow
            inflow = link_obs[0]          # inflow
            current_width = link_obs[-1]

            # Calculate the change in width based on pressure differential
            # change_in_width = self.K * (p_up - self.W_backpressure * p_down)
            change_in_width = 1
            # if density >= 3.5:
            #     new_target_width = self.outgoing_links[i].width
            if density > self.threshold_density:
                new_target_width = current_width + change_in_width
            elif density > self.threshold_density:
                new_target_width = current_width - change_in_width
            else:
                # Keep gate open to max width (action space high bound)
                new_target_width = self.outgoing_links[i].width
            # new_target_width = self.outgoing_links[i].width
            actions.append(new_target_width)
            # actions.append(self.outgoing_links[i].width)
        # actions[1] = 2
        # actions[3] = 2.8

        return np.array(actions, dtype=np.float32)

class RuleBasedSeparatorAgent(BaseAgent):
    """
    A rule-based agent for controlling separators based on in/outflow balance.
    
    Optionally uses a moving average buffer to smooth instantaneous inflow measurements,
    reducing sensitivity to fluctuations.
    """
    def __init__(self, width: float, use_smoothing: bool = False, buffer_size: int = 5):
        """
        Initializes the RuleBasedSeparatorAgent.

        Args:
            width (float): The width of the road
            use_smoothing (bool): If True, uses moving average smoothing on inflows. Default: False
            buffer_size (int): Window size for the moving average buffer. Only used if use_smoothing=True. Default: 5
        """
        self.road_width = width
        self.use_smoothing = use_smoothing
        self.buffer_size = buffer_size
        
        # Initialize inflow buffers for each direction (only if smoothing is enabled)
        if self.use_smoothing:
            self._link_inflow_buffer = []
            self._reversed_link_inflow_buffer = []
        else:
            self._link_inflow_buffer = None
            self._reversed_link_inflow_buffer = None

    def _update_and_smooth_inflow(self, buffer: list, current_inflow: float) -> float:
        """
        Update the inflow buffer and return the smoothed (moving average) inflow.
        
        Args:
            buffer (list): The inflow history buffer
            current_inflow (float): Current instantaneous inflow value
            
        Returns:
            float: Smoothed inflow value (moving average)
        """
        if not self.use_smoothing:
            return current_inflow
        
        buffer.append(current_inflow)
        
        # Keep buffer size fixed
        if len(buffer) > self.buffer_size:
            buffer.pop(0)
        
        # Return moving average
        return float(np.mean(buffer))

    def take_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Calculates actions based on the inflows of the link and reversed link.
        
        If smoothing is enabled, uses moving average of historical inflows.
        Otherwise, uses instantaneous inflow values.
        """
        # Extract raw inflows
        # Observation structure: [inflow, outflow, reverse_inflow, reverse_outflow] (or with density if enabled)
        link_inflow_raw = obs[1] if len(obs) > 1 else 0.0
        reversed_link_inflow_raw = obs[4] if len(obs) > 4 else 0.0
        
        # Apply smoothing if enabled
        if self.use_smoothing:
            link_inflow = self._update_and_smooth_inflow(self._link_inflow_buffer, link_inflow_raw)
            reversed_link_inflow = self._update_and_smooth_inflow(self._reversed_link_inflow_buffer, reversed_link_inflow_raw)
        else:
            link_inflow = link_inflow_raw
            reversed_link_inflow = reversed_link_inflow_raw
        
        # Allocate width proportionally to inflows
        if link_inflow + reversed_link_inflow == 0:
            actions = self.road_width / 2  # if the inflow is 0, set the action to the middle of the road
        else:
            actions = self.road_width * link_inflow / (link_inflow + reversed_link_inflow)
        return np.array([actions], dtype=np.float32)

if __name__ == "__main__":
    from rl.pz_pednet_env import PedNetParallelEnv
    dataset = "45_intersections"
    # dataset = "small_network"
    env = PedNetParallelEnv(dataset, obs_mode="option2", action_gap=1, render_mode="animate", verbose=True)

    #create a rule-based agents
    rule_based_gater_agents = {}
    rule_based_separator_agents = {}
    for agent_id in env.agent_manager.get_gater_agents():
        rule_based_gater_agents[agent_id] = RuleBasedGaterAgent(env.agent_manager.get_gater_outgoing_links(agent_id), env.obs_mode, threshold_density=3)
    for agent_id in env.agent_manager.get_separator_agents():
        rule_based_separator_agents[agent_id] = RuleBasedSeparatorAgent(env.agent_manager.get_separator_links(agent_id)[0].width, use_smoothing=True, buffer_size=5)

    episode_rewards = {agent_id: 0.0 for agent_id in env.agents}
    observations, infos = env.reset(seed=3, options={"randomize": False})
    # observations, infos = env.reset(seed=42, options={"randomize": True})
    # observations, infos = env.reset(seed=43, options={"randomize": True})
    # for step in range(env.simulation_steps):
    done = False
    while not done:
        actions = {}
        for agent_id in env.agents:
            if agent_id in rule_based_gater_agents:
                actions[agent_id] = rule_based_gater_agents[agent_id].take_action(observations[agent_id])
            elif agent_id in rule_based_separator_agents:
                actions[agent_id] = rule_based_separator_agents[agent_id].take_action(observations[agent_id])
            # print(actions[agent_id])
            # if agent_id == "gate_24":
            #     print(f"Step {step}: Agent {agent_id} action: {actions[agent_id]}")
        #         pass
        observations, rewards, terminations, truncations, infos = env.step(actions)
        done = any(terminations.values()) or any(truncations.values())
        for agent_id in env.agents:
            episode_rewards[agent_id] += rewards[agent_id]

    # final rewards
    for agent_id in env.possible_agents:
        print(f"Agent {agent_id} final reward: {episode_rewards[agent_id]}")
    # total reward
    avg_reward = np.mean(list(episode_rewards.values()))
    print(f"Total reward: {avg_reward}")

    env.save(simulation_dir="../../outputs/rule_based_agents")
    env.render(simulation_dir="../../outputs/rule_based_agents", variable='density', vis_actions=True, save_dir=None)


