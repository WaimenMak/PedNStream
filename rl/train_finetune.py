# -*- coding: utf-8 -*-
"""
Fine-tune PPO HRL agents on Scenario F using a pretrained checkpoint.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import torch
from rl import PedNetParallelEnv
from rl.rl_utils import RunningNormalizeWrapper, evaluate_agents, load_all_agents
from rl.agents.PPO_hrl import PPOAgentHRL, train_hrl_multi_agent_batch

if __name__ == "__main__":
    algo = "ppo_hrl"
    SEED = 77
    NORM = False   
    builder_norm_obs = False  
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
    print(f"Fine-tuning {algo} Agents on PedNet Environment (Scenario F)")
    print("=" * 60)

    dataset = "butterfly_scF"

    # Create environment with normalization wrapper
    base_env = PedNetParallelEnv(
        dataset=dataset, normalize_obs=builder_norm_obs, obs_mode=STATE_OPTION, render_mode="animate", action_gap=action_gap
    )
    env = RunningNormalizeWrapper(base_env, norm_obs=NORM, norm_reward=norm_ret)
    env.seed(SEED)

    agents = {}
    for agent_id in env.possible_agents:
        agents[agent_id] = PPOAgentHRL(
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
            num_lstm_layers=2,
            num_heads=2,
            use_param_noise=False,
            use_action_noise=False,
            num_episodes=400,
            tm_window=20,
            max_duration=7,
            duration_entropy_coef=0.05,
            duration_entropy_coef_min=0.001,
            value_fusion='mean',
        )

    # Load pretrained checkpoints
    checkpoint_path = project_root / "rl/analysis/checkpoints/best_checkpoints/checkpoint.pt"
    print(f"Loading pretrained weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract the source agent's state dict (either 'gate_2' or whatever is available)
    source_agent_state = checkpoint.get("gate_2")
    if source_agent_state is None:
        source_agent_key = list(checkpoint.keys())[0] if checkpoint else None
        if source_agent_key and isinstance(checkpoint[source_agent_key], dict) and 'actor_state_dict' in checkpoint[source_agent_key]:
            source_agent_state = checkpoint[source_agent_key]
            
    if source_agent_state:
        for agent_id, agent in agents.items():
            agent.actor.load_state_dict(source_agent_state['actor_state_dict'])
            agent.value_net.load_state_dict(source_agent_state['critic_state_dict'])
            print(f"Loaded pretrained weights for agent: {agent_id}")
    else:
        print("Warning: Could not find valid pretrained weights in checkpoint.")

    # Train PPO HRL agents
    # return_dict, _ = train_hrl_multi_agent_batch(
    #     env, agents, num_episodes=500, num_trajectories_per_update=2, delta_actions=True,
    #     randomize=randomize, agents_saved_dir=f"./checkpoints/ppo_hrl_finetune_{dataset}",
    #     num_val_episodes=10, val_freq=10, use_wandb=True,
    #     debug_save_dir=f"rl_training/{dataset}/ppo_hrl_finetune_debug",
    #     debug_save_episodes=[5, 50, 100, 200, 500]
    # )
    #
    # # Evaluation phase
    # SEED = 42
    # randomized = True
    # num_runs = 15
    #
    # # Load the best fine-tuned agents
    # agents, config_data = load_all_agents(save_dir=f"./checkpoints/ppo_hrl_finetune_{dataset}", device="cpu")
    #
    # base_env = PedNetParallelEnv(
    #     dataset=dataset, normalize_obs=builder_norm_obs, obs_mode=STATE_OPTION, render_mode="animate"
    # )
    # env = RunningNormalizeWrapper(base_env, norm_obs=NORM, norm_reward=False, training=False)
    #
    # if 'normalization_stats' in config_data:
    #     env.set_normalization_stats(config_data['normalization_stats'])
    #
    # rl_results = evaluate_agents(
    #     env, agents,
    #     delta_actions=True,
    #     deterministic=True,
    #     seed=SEED,
    #     randomize=randomized,
    #     num_runs=num_runs,
    #     save_dir=f"rl_training/{dataset}/ppo_hrl_finetune"
    # )
    #
    # # Compare with no control
    # env = PedNetParallelEnv(
    #     dataset=dataset, normalize_obs=False, obs_mode="option2", render_mode="animate", action_gap=action_gap
    # )
    # no_control_agents = {agent_id: None for agent_id in env.possible_agents}
    # no_control_results = evaluate_agents(
    #     env, no_control_agents,
    #     delta_actions=False,
    #     seed=SEED,
    #     no_control=True,
    #     randomize=randomized,
    #     num_runs=num_runs,
    #     save_dir=f"rl_training/{dataset}/no_control"
    # )
    #
    # print("\n" + "=" * 60)
    # print("Comparison of All Methods")
    # print("=" * 60)
    # print(f"Fine-tuned {algo} avg reward: {rl_results['avg_reward']:.3f}")
    # print(f"No control avg reward:        {no_control_results['avg_reward']:.3f}")
    # print("=" * 60)

    # render the results
    project_root = Path(__file__).resolve().parent.parent
    # Render final simulation
    env.render(
        simulation_dir=str(project_root / f"outputs/rl_training/{dataset}/ppo_hrl_finetune_run5"),
        # simulation_dir=str(project_root / f"outputs/rl_training/{dataset}/no_control"),
        variable='density',
        vis_actions=True,
        save_dir=None
    )
