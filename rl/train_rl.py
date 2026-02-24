# -*- coding: utf-8 -*-
"""
Train PPO agents on PedNetParallelEnv using independent learning.

This script trains each agent independently using PPO. Each agent observes
its local state and learns to control its gate/separator to minimize congestion.
"""

import sys
from pathlib import Path

from torch.backends.cudnn import deterministic

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
from rl.agents.PPO_backup import PPOAgent, train_on_policy_multi_agent
from rl.agents.SAC_copy import SACAgent, train_off_policy_multi_agent
from rl.agents.PPO_tbptt import PPOAgent as PPOAgent_tbptt, train_on_policy_multi_agent_batch
from rl.agents.PPO_dyna import PPOAgent_dyna
from rl.agents.POME import POMEAgent
from rl.agents.PPO_hrl import PPOAgentHRL, train_hrl_multi_agent_batch
from rl.agents.rule_based import RuleBasedGaterAgent
from rl.agents.optimization_based import DecentralizedOptimizationAgent

if __name__ == "__main__":
    """ option1: inoutflow of the node and gate widths
        option2: density on outgoing links and gate widths
        option3:inoutflow of the node and other side, gate widths
    """
    algo = "ppo_hrl"  # "ppo", "ppo_dyna", "ppo_tbptt", "pome", "sac"
    SEED = 77
    NORM = False   # running mean-std normalization for observations
    builder_norm_obs = False  # normalize observations in the environment builder
    STATE_OPTION = "option4"
    randomize = True
    norm_ret = True
    action_gap = 1
    # set torch seed
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    # Enable deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("=" * 60)
    print(f"Training {algo} Agents on PedNet Environment")
    print("=" * 60)

    # dataset = "45_intersections"
    # dataset = "two_coordinators"
    # dataset = "one_intersection_v0"
    # dataset = "small_network"
    dataset = "butterfly_scF"

    # Create environment with normalization wrapper
    base_env = PedNetParallelEnv(
        dataset=dataset, normalize_obs=builder_norm_obs, obs_mode=STATE_OPTION, render_mode="animate", action_gap=action_gap
    )
    env = RunningNormalizeWrapper(base_env, norm_obs=NORM, norm_reward=norm_ret)
    env.seed(SEED)

    if algo == "ppo":
        # Create agents (all use stateful LSTM)
        agents = {agent_id: PPOAgent(
            obs_dim=env.observation_space(agent_id).shape[0],
            act_dim=env.action_space(agent_id).shape[0],
            act_low=env.action_space(agent_id).low,
            act_high=env.action_space(agent_id).high,
            actor_lr=9e-5,
            critic_lr=2e-4,
            gamma=0.99,
            lmbda=0.96,
            entropy_coef=0.01,
            kl_tolerance=0.01,
            use_delta_actions=True,
            max_delta=2.5,
            lstm_hidden_size=64,
            num_lstm_layers=1,
            use_gat_lstm=False,
            gat_hidden_size=64,
            use_stacked_obs=False,
            stack_size=0,
            hidden_size=64,
            kernel_size=4,
            use_param_noise=False,
            use_action_noise=False,
            total_updates=100*0.8,
            param_noise_std_min=0
        ) for agent_id in env.possible_agents}

        # Train PPO agents
        return_dict, _ = train_on_policy_multi_agent(
            env, agents, num_episodes=100, delta_actions=True,
            randomize=randomize,
            agents_saved_dir=f"./checkpoints/ppo_agents_{dataset}",
            num_val_episodes=5,
            val_freq=10,
            use_wandb=True
        )

    elif algo == "ppo_dyna":
        agents = {agent_id: PPOAgent_dyna(
            obs_dim=env.observation_space(agent_id).shape[0],
            act_dim=env.action_space(agent_id).shape[0],
            act_low=env.action_space(agent_id).low,
            act_high=env.action_space(agent_id).high,
            actor_lr=9e-5,
            critic_lr=2e-4,
            gamma=0.99,
            lmbda=0.96,
            entropy_coef=0.01,
            kl_tolerance=0.01,
            entropy_coef_min=0,
            use_delta_actions=True,
            max_delta=2.5,
            lstm_hidden_size=64,
            num_lstm_layers=1,
            use_param_noise=False,
            use_action_noise=False,
            num_episodes=200,
            tm_window=50,  # truncate backpropagation window
            use_dyna=False,
            model_lr=3e-4,
            dream_rollouts=20,
            dream_horizon=5,

        ) for agent_id in env.possible_agents}

        # agents, config_data = load_all_agents(save_dir=f"./checkpoints/{algo}_agents_butterfly_scA", device="cpu")
        return_dict, _ = train_on_policy_multi_agent(
            env, agents, num_episodes=200, delta_actions=True,
            randomize=randomize, agents_saved_dir=f"./checkpoints/ppo_dyna_agents_{dataset}",
            num_val_episodes=5, val_freq=10, use_wandb=False
        )

    elif algo == "ppo_tbptt":
        agents = {agent_id: PPOAgent_tbptt(
            obs_dim=env.observation_space(agent_id).shape[0],
            act_dim=env.action_space(agent_id).shape[0],
            act_low=env.action_space(agent_id).low,
            act_high=env.action_space(agent_id).high,
            actor_lr=1e-4,
            critic_lr=2e-4,
            use_lr_decay=True,
            gamma=0.99,
            lmbda=0.96,
            entropy_coef=0.04,
            kl_tolerance=0.02,
            entropy_coef_min=0.001,
            use_delta_actions=True,
            max_delta=2.5,
            lstm_hidden_size=64,
            num_lstm_layers=1,
            use_param_noise=False,
            use_action_noise=False,
            num_episodes=200,
            tm_window=50,  # truncate backpropagation window
            epochs=10

        ) for agent_id in env.possible_agents}

        # agents, config_data = load_all_agents(save_dir=f"./checkpoints/{algo}_agents_butterfly_scA", device="cpu")
        # return_dict, _ = train_on_policy_multi_agent(
        #     env, agents, num_episodes=200, delta_actions=True,
        #     randomize=randomize, agents_saved_dir=f"./checkpoints/ppo_tbptt_agents_{dataset}",
        #     num_val_episodes=5, val_freq=10, use_wandb=False
        # )
        # env.set_training(True)
        return_dict, _ = train_on_policy_multi_agent_batch(
            env, agents, num_episodes=200, num_trajectories_per_update=2, delta_actions=True,
            randomize=randomize, agents_saved_dir=f"./checkpoints/ppo_tbptt_agents_{dataset}",
            num_val_episodes=10, val_freq=10, use_wandb=True,
            debug_save_dir=f"rl_training/{dataset}/ppo_tbptt_debug",
            debug_save_episodes=[5, 50, 100, 150, 200]
        )
    elif algo == "ppo_hrl":
        agents = {agent_id: PPOAgentHRL(
            obs_dim=env.observation_space(agent_id).shape[0],
            act_dim=env.action_space(agent_id).shape[0],
            act_low=env.action_space(agent_id).low,
            act_high=env.action_space(agent_id).high,
            actor_lr=1e-4,
            critic_lr=2e-4,
            use_lr_decay=False,
            gamma=0.99,
            lmbda=0.96,
            entropy_coef=0.04,
            kl_tolerance=0.02,
            use_delta_actions=True,
            max_delta=2.5,
            lstm_hidden_size=64,
            num_lstm_layers=1,
            num_heads=2,
            use_param_noise=False,
            use_action_noise=False,
            num_episodes=300,
            tm_window=20,
            max_duration=7,
            duration_entropy_coef=0.05,
            duration_entropy_coef_min=0.001,
            value_fusion='mean',
        ) for agent_id in env.possible_agents}
        # agents, config_data = load_all_agents(save_dir=f"./checkpoints/{algo}_agents_butterfly_scB", device="cpu")
        return_dict, _ = train_hrl_multi_agent_batch(
            env, agents, num_episodes=300, num_trajectories_per_update=2, delta_actions=True,
            randomize=randomize, agents_saved_dir=f"./checkpoints/ppo_hrl_agents_{dataset}",
            num_val_episodes=10, val_freq=10, use_wandb=True,
            debug_save_dir=f"rl_training/{dataset}/ppo_hrl_debug",
            debug_save_episodes=[5, 50, 100, 150, 200]
        )
    elif algo == "pome":
        agents = {agent_id: POMEAgent(
            obs_dim=env.observation_space(agent_id).shape[0],
            act_dim=env.action_space(agent_id).shape[0],
            act_low=env.action_space(agent_id).low,
            act_high=env.action_space(agent_id).high,
            actor_lr=9e-5,
            critic_lr=2e-4,
            gamma=0.99,
            lmbda=0.96,
            entropy_coef=0.01,
            kl_tolerance=0.01,
            use_delta_actions=True,
            max_delta=2.5,
            lstm_hidden_size=64,
            num_lstm_layers=1,
            use_param_noise=False,
            use_action_noise=False,
            num_episodes=100,
            tm_window=50,
            model_lr=5e-4,
            # alpha=0.1,
            # alpha_min=0.0,
            norm_reward=True,
        ) for agent_id in env.possible_agents}
        # agents, config_data = load_all_agents(save_dir=f"./checkpoints/{algo}_agents_butterfly_scA", device="cpu")
        return_dict, _ = train_on_policy_multi_agent(
            env, agents, num_episodes=100, delta_actions=True,
            randomize=randomize, agents_saved_dir=f"./checkpoints/pome_agents_{dataset}",
            num_val_episodes=5, val_freq=10, use_wandb=False
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
            hidden_size=64,
            kernel_size=4,
            stack_size=3,
            actor_lr=1e-5,
            critic_lr=3e-4,
        ) for agent_id in env.possible_agents}

        return_dict, _ = train_off_policy_multi_agent(
            env, agents, num_episodes=100, delta_actions=True,
            randomize=randomize,
            agents_saved_dir=f"./checkpoints/sac_agents_{dataset}",
            use_wandb=False
        )

    # plot critic loss
    # import matplotlib.pyplot as plt
    # plt.plot(agents["gate_24"].critic_loss_history)
    # plt.title("Critic Loss")
    # plt.xlabel("Update")
    # plt.ylabel("Loss")
    # plt.show()

    # Save agents with normalization stats
    # save_all_agents(
    #     agents,
    #     save_dir=f"{algo}_agents_{dataset}",
    #     metadata={'dataset': dataset, 'num_episodes': 60},
    #     normalization_stats=env.get_normalization_stats()
    # )

    SEED = 42
    randomized = True
    num_runs = 15
    # Load agents for evaluation
    agents, config_data = load_all_agents(save_dir=f"./checkpoints/{algo}_agents_{dataset}", device="cpu")
    # agents, config_data = load_all_agents(save_dir=f"ppo_agent_best", device="cpu")

    # Create fresh environment for evaluation
    base_env = PedNetParallelEnv(
        dataset=dataset, normalize_obs=builder_norm_obs, obs_mode=STATE_OPTION, render_mode="animate"
    )
    env = RunningNormalizeWrapper(base_env, norm_obs=NORM, norm_reward=False, training=False)

    # Restore normalization stats for normed obs
    if 'normalization_stats' in config_data:
        env.set_normalization_stats(config_data['normalization_stats'])

    # Evaluate RL agents
    rl_results = evaluate_agents(
        env, agents,
        delta_actions=True,
        deterministic=True,
        seed=SEED,
        randomize=randomized,
        num_runs=num_runs,
        save_dir=f"rl_training/{dataset}/{algo}"
    )

    # Evaluate rule-based agents
    env = PedNetParallelEnv(
        dataset=dataset, normalize_obs=False, obs_mode="option2", render_mode="animate", action_gap=action_gap
    )
    rule_based_agents = {agent_id: RuleBasedGaterAgent(env.agent_manager.get_gater_outgoing_links(agent_id), env.obs_mode, threshold_density=3) for agent_id in env.agent_manager.get_gater_agents()}
    # For rule-based agents (from rule_based.py)
    rule_results = evaluate_agents(
        env, rule_based_agents,
        delta_actions=False,
        seed=SEED,  # Same seed for fair comparison
        randomize=randomized,
        num_runs=num_runs,
        save_dir=f"rl_training/{dataset}/rule_based"
    )

    # # Evaluate optimization-based agents
    # env = PedNetParallelEnv(
    #     dataset=dataset, normalize_obs=False, obs_mode="option2", render_mode="animate", action_gap=action_gap
    # )
    # optimization_based_agents = {agent_id: DecentralizedOptimizationAgent(env.network, env.agent_manager, agent_id=agent_id, verbose=False) for agent_id in env.agent_manager.get_gater_agents()}
    # optimization_based_results = evaluate_agents(
    #     env, optimization_based_agents,
    #     delta_actions=False,
    #     seed=SEED,  # Same seed for fair comparison
    #     randomize=randomized,
    #     num_runs=3,
    #     save_dir=f"rl_training/{dataset}/optimization_based"
    # )

    # no control policy
    env = PedNetParallelEnv(
        dataset=dataset, normalize_obs=False, obs_mode="option2", render_mode="animate", action_gap=action_gap
    )
    no_control_agents = {agent_id: None for agent_id in env.possible_agents}
    no_control_results = evaluate_agents(
        env, no_control_agents,
        delta_actions=False,
        seed=SEED,  # Same seed for fair comparison
        no_control=True,
        randomize=randomized,
        num_runs=num_runs,
        save_dir=f"rl_training/{dataset}/no_control"
    )

    # Compare results
    print("\n" + "=" * 60)
    print("Comparison of All Methods")
    print("=" * 60)
    print(f"{algo} avg reward:        {rl_results['avg_reward']:.3f}")
    print(f"Rule-based avg reward: {rule_results['avg_reward']:.3f}")
    # print(f"Optimization-based avg reward: {optimization_based_results['avg_reward']:.3f}")
    print(f"No control avg reward: {no_control_results['avg_reward']:.3f}")
    print("=" * 60)
    #
    # #evaluation metrics
    # #ppo
    # from rl.rl_utils import compute_network_congestion_metric
    # rl_throughput = compute_network_congestion_metric(simulation_dir=f"../outputs/rl_training/{dataset}/{algo}", threshold_ratio=0.65)
    # print(f"{algo} congestion time: {rl_throughput['congestion_time']:.3f}")
    # #rule-based
    # rule_throughput = compute_network_congestion_metric(simulation_dir=f"../outputs/rl_training/{dataset}/rule_based", threshold_ratio=0.65)
    # print(f"Rule-based congestion time: {rule_throughput['congestion_time']:.3f}")
    # #no control
    # no_control_throughput = compute_network_congestion_metric(simulation_dir=f"../outputs/rl_training/{dataset}/no_control", threshold_ratio=0.65)
    # print(f"No control congestion time: {no_control_throughput['congestion_time']:.3f}")


    # Render final simulation
    env.render(
        simulation_dir=str(project_root / f"outputs/rl_training/{dataset}/{algo}_run5"),
        # simulation_dir=str(project_root / f"outputs/rl_training/{dataset}/no_control"),
        variable='density',
        vis_actions=True,
        save_dir=None
    )
