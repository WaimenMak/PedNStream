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
    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        Takes an observation and returns an action.
        """
        pass


class RuleBasedGaterAgent(BaseAgent):
    """
    A rule-based agent for controlling gaters based on pressure differential.

    This agent implements an "Incremental Pressure Balancing" algorithm. It adjusts
    the gate width based on the difference between upstream pressure (demand) and
    downstream back-pressure (congestion).
    """

    def __init__(self, num_outgoing_links: int, with_density_obs: bool, K: float = 0.1, W_backpressure: float = 1.5):
        """
        Initializes the RuleBasedGaterAgent.

        Args:
            num_outgoing_links (int): The number of outgoing links this gater controls.
            with_density_obs (bool): Whether the observation includes density.
                                     This agent's logic requires density.
            K (float): Responsiveness parameter determining how fast the gate reacts.
            W_backpressure (float): Weight for downstream back-pressure. A higher
                                    value makes the agent more cautious about congestion.
        """
        if not with_density_obs:
            raise ValueError("RuleBasedGaterAgent requires density information ('with_density_obs' must be True).")

        self.num_outgoing_links = num_outgoing_links
        self.K = K
        self.W_backpressure = W_backpressure
        self.features_per_link = 4  # density, inflow, reverse_outflow, current_width

    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        Calculates actions based on the incremental pressure balancing algorithm.

        Args:
            obs (np.ndarray): The observation for this agent. It's a flattened
                              array of features for each outgoing link, padded to
                              the max out-degree in the network.

        Returns:
            np.ndarray: An array of target gate widths for each outgoing link.
        """
        actions = []
        for i in range(len(self.num_outgoing_links)):
            start_idx = i * self.features_per_link
            link_obs = obs[start_idx: start_idx + self.features_per_link]

            # The observation vector is structured as:
            # [density, inflow, reverse_outflow, current_width]
            p_down = link_obs[0]        # density
            p_up = link_obs[2]          # reverse_link.outflow
            current_width = link_obs[3]

            # Calculate the change in width based on pressure differential
            change_in_width = self.K * (p_up - self.W_backpressure * p_down)
            new_target_width = current_width + change_in_width
            actions.append(new_target_width)

        return np.array(actions, dtype=np.float32)

class RuleBasedSeparatorAgent(BaseAgent):
    """
    A rule-based agent for controlling separators based on in/outflow balance.
    """
    def __init__(self, width: float):
        """
        Initializes the RuleBasedSeparatorAgent.
        Args:
            width: The width of the road
        """
        self.road_width = width

    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        Calculates actions based on the inflows of the link and revsersed link.
        """
        link_inflow = obs[1]
        reversed_link_inflow = obs[4]
        if link_inflow + reversed_link_inflow == 0:
            actions = self.road_width / 2 # if the inflow is 0, set the action to the middle of the road
        else:
            actions = self.road_width * link_inflow / (link_inflow + reversed_link_inflow)
        return np.array([actions], dtype=np.float32)

if __name__ == "__main__":
    from src.rl.pz_pednet_env import PedNetParallelEnv
    dataset = "long_corridor"
    env = PedNetParallelEnv(dataset, with_density_obs=True)

    #create a rule-based agents
    rule_based_gater_agents = {}
    rule_based_separator_agents = {}
    for agent_id in env.agent_manager.get_gater_agents():
        rule_based_gater_agents[agent_id] = RuleBasedGaterAgent(env.agent_manager.get_gater_outgoing_links(agent_id), env.with_density_obs)
    for agent_id in env.agent_manager.get_separator_agents():
        rule_based_separator_agents[agent_id] = RuleBasedSeparatorAgent(env.agent_manager.get_separator_links(agent_id)[0].width)
    observations, infos = env.reset()
    for step in range(env.simulation_steps):
        actions = {}
        for agent_id in env.agents:
            if agent_id in rule_based_gater_agents:
                actions[agent_id] = rule_based_gater_agents[agent_id].act(observations[agent_id])
            elif agent_id in rule_based_separator_agents:
                actions[agent_id] = rule_based_separator_agents[agent_id].act(observations[agent_id])
            print(actions[agent_id])
        #     if actions[agent_id] == 0:
        #         pass
        observations, rewards, terminations, truncations, infos = env.step(actions)
        # print(rewards)

    env.save(simulation_dir="rule_based_agents")
    env.render(mode="animate", simulation_dir="../../outputs/rule_based_agents", variable='density', vis_actions=True)


