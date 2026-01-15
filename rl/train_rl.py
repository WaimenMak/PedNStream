# -*- coding: utf-8 -*-
"""
Train PPO agents on PedNetParallelEnv using independent learning.

This script trains each agent independently using PPO. Each agent observes
its local state and learns to control its gate/separator to minimize congestion.
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import numpy as np
from collections import deque
from rl import PedNetParallelEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from rl.rl_utils import (
    RunningNormalizeWrapper,
    save_all_agents,
    load_all_agents,
    evaluate_agents,
)
from rl.agents.PPO import PPOAgent, train_on_policy_multi_agent
from rl.agents.SAC import SACAgent, train_off_policy_multi_agent
from rl.agents.rule_based import RuleBasedGaterAgent


if __name__ == "__main__":
    """ option1: inoutflow of the node and gate widths
        option2: density on outgoing links and gate widths
        option3:inoutflow of the node and other side, gate widths
    """
    algo = "sac"
    print("=" * 60)
    print(f"Training {algo} Agents on PedNet Environment")
    print("=" * 60)
    
    dataset = "45_intersections"
    
    # Create environment with normalization wrapper
    base_env = PedNetParallelEnv(
        dataset=dataset, normalize_obs=False, obs_mode="option3", render_mode="animate"
    )
    env = RunningNormalizeWrapper(base_env, norm_obs=False, norm_reward=True)

    if algo == "ppo":
        # Create agents (all use stateful LSTM)
        agents = {agent_id: PPOAgent(
            obs_dim=env.observation_space(agent_id).shape[0],
            act_dim=env.action_space(agent_id).shape[0],
            act_low=env.action_space(agent_id).low,
            act_high=env.action_space(agent_id).high,
            actor_lr=2e-4,
            critic_lr=3e-4,
            # gamma=0.98,
            lmbda=0.96,
            entropy_coef=0.01,
            kl_tolerance=0.01,
            use_delta_actions=True,
            max_delta=2.5,
            # lstm_hidden_size=64,
            # num_lstm_layers=1,
            use_stacked_obs=False,
            stack_size=5,
            hidden_size=64,
            kernel_size=4,
        ) for agent_id in env.possible_agents}
        # Train PPO agents
        return_dict, _ = train_on_policy_multi_agent(
            env, agents, num_episodes=100, delta_actions=True,
            randomize=False,
            seed=None,
            agents_saved_dir=f"ppo_agents_{dataset}",
        )
    elif algo == "sac":
        agents = {agent_id: SACAgent(
            obs_dim=env.observation_space(agent_id).shape[0],
            act_dim=env.action_space(agent_id).shape[0],
            act_low=env.action_space(agent_id).low,
            act_high=env.action_space(agent_id).high,
            target_entropy=-env.action_space(agent_id).shape[0],
            buffer_size=100000,
            max_delta=2.5,
            hidden_size=32,
            kernel_size=4,
            stack_size=10,
            actor_lr=3e-4,
            critic_lr=3e-3,
        ) for agent_id in env.possible_agents}

        return_dict, _ = train_off_policy_multi_agent(
            env, agents, num_episodes=100, delta_actions=True,
            randomize=False,
            seed=None,
            agents_saved_dir=f"sac_agents_{dataset}",
        )

    # plot critic loss
    import matplotlib.pyplot as plt
    plt.plot(agents["gate_24"].critic_loss_history)
    plt.title("Critic Loss")
    plt.xlabel("Update")
    plt.ylabel("Loss")
    plt.show()

    # Save agents with normalization stats
    # save_all_agents(
    #     agents,
    #     save_dir=f"{algo}_agents_{dataset}",
    #     metadata={'dataset': dataset, 'num_episodes': 60},
    #     normalization_stats=env.get_normalization_stats()
    # )

    SEED = 50
    randomized = False
    # Load agents for evaluation
    agents, config_data = load_all_agents(save_dir=f"{algo}_agents_{dataset}", device="cpu")
    # ppo_agents, config_data = load_all_agents(save_dir=f"ppo_agent_best", device="cpu")

    # Create fresh environment for evaluation
    base_env = PedNetParallelEnv(
        dataset=dataset, normalize_obs=False, obs_mode="option3", render_mode="animate"
    )
    env = RunningNormalizeWrapper(base_env, norm_obs=False, norm_reward=False, training=False)

    # Restore normalization stats
    if 'normalization_stats' in config_data:
        env.set_normalization_stats(config_data['normalization_stats'])

    # Evaluate RL agents
    rl_results = evaluate_agents(
        env, agents,
        delta_actions=True,
        deterministic=True,
        seed=SEED,
        randomize=randomized,
        save_dir=f"rl_training/{dataset}/{algo}"
    )

    # Evaluate rule-based agents
    env = PedNetParallelEnv(
        dataset=dataset, normalize_obs=False, obs_mode="option2", render_mode="animate"
    )
    rule_based_agents = {agent_id: RuleBasedGaterAgent(env.agent_manager.get_gater_outgoing_links(agent_id), env.obs_mode, threshold_density=3) for agent_id in env.agent_manager.get_gater_agents()}
    # For rule-based agents (from rule_based.py)
    rule_results = evaluate_agents(
        env, rule_based_agents,
        delta_actions=False,
        seed=SEED,  # Same seed for fair comparison
        randomize=randomized,
        save_dir=f"rl_training/{dataset}/rule_based"
    )

    # no control policy
    env = PedNetParallelEnv(
        dataset=dataset, normalize_obs=False, obs_mode="option2", render_mode="animate"
    )
    no_control_agents = {agent_id: None for agent_id in env.possible_agents}
    no_control_results = evaluate_agents(
        env, no_control_agents,
        delta_actions=False,
        seed=SEED,  # Same seed for fair comparison
        no_control=True,
        randomize=randomized,
        save_dir=f"rl_training/{dataset}/no_control"
    )

    # Compare results
    print("\n" + "=" * 60)
    print("Comparison of All Methods")
    print("=" * 60)
    print(f"{algo} avg reward:        {rl_results['avg_reward']:.3f}")
    print(f"Rule-based avg reward: {rule_results['avg_reward']:.3f}")
    print(f"No control avg reward: {no_control_results['avg_reward']:.3f}")
    print("=" * 60)
    #
    # #evaluation metrics
    #ppo
    from rl.rl_utils import compute_network_congestion_metric
    rl_throughput = compute_network_congestion_metric(simulation_dir=f"../outputs/rl_training/{dataset}/{algo}", threshold_ratio=0.65)
    print(f"{algo} congestion time: {rl_throughput['congestion_time']:.3f}")
    #rule-based
    rule_throughput = compute_network_congestion_metric(simulation_dir=f"../outputs/rl_training/{dataset}/rule_based", threshold_ratio=0.65)
    print(f"Rule-based congestion time: {rule_throughput['congestion_time']:.3f}")
    #no control
    no_control_throughput = compute_network_congestion_metric(simulation_dir=f"../outputs/rl_training/{dataset}/no_control", threshold_ratio=0.65)
    print(f"No control congestion time: {no_control_throughput['congestion_time']:.3f}")


    # Render final simulation
    env.render(
        simulation_dir=str(project_root / f"outputs/rl_training/{dataset}/{algo}"),
        variable='density',
        vis_actions=True,
        save_dir=None
    )
