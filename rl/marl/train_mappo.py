# -*- coding: utf-8 -*-
"""
Fine-tune PPO HRL agents on Scenario F using a pretrained checkpoint.
"""

# Find the project root (3 levels up from rl/marl/train_mappo.py)
from pathlib import Path
import sys
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import torch
import numpy as np
from tqdm import tqdm
from rl import PedNetParallelEnv
from rl.rl_utils import RunningNormalizeWrapper, validate_and_save_best, load_all_agents, evaluate_agents
from rl.marl.MAPPO_hrl import MAPPOAgentHRL


try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def train_mappo_batch(env, agents, delta_actions=False, num_episodes=50,
                      num_trajectories_per_update=4, randomize=False,
                      agents_saved_dir=None, use_wandb=True,
                      val_freq=10, num_val_episodes=3,
                      debug_save_dir=None, debug_save_episodes=None):
    """
    Train MAPPO agents with duration-augmented actions.
    
    In MAPPO, the actor is decentralized (uses local state), but the critic is centralized
    (uses a concatenated global state). The global reward is the sum of local rewards.
    """
    if use_wandb and WANDB_AVAILABLE:
        if wandb.run is None:
            wandb.init(project="crowd-control-rl", name="mappo-hrl-training")

    return_dict = {agent_id: [] for agent_id in agents.keys()}
    global_episode = 0
    global_update = 0
    best_avg_return = float('-inf')

    debug_episodes = tuple(debug_save_episodes) if debug_save_episodes else None

    # Initialize batch buffers
    for agent in agents.values():
        agent.init_batch_buffer()

    # Adjust total_updates for batch training
    first_agent = next(iter(agents.values()))
    if hasattr(first_agent, "total_updates"):
        effective_updates = max(
            1,
            int(num_episodes / float(max(1, num_trajectories_per_update)) * 0.8),
        )
        for agent in agents.values():
            agent.total_updates = effective_updates

    # Tracking
    batch_returns = {aid: [] for aid in agents.keys()}
    batch_true_returns = {aid: [] for aid in agents.keys()}
    batch_policy_mu = {aid: [] for aid in agents.keys()}
    batch_policy_sigma = {aid: [] for aid in agents.keys()}
    batch_duration_probs = {aid: [] for aid in agents.keys()}
    batch_sampled_durations = {aid: [] for aid in agents.keys()}

    num_iterations = 10
    episodes_per_iteration = num_episodes // num_iterations

    # Make agent ordering fixed for concatenated state
    agent_keys = sorted(list(agents.keys()))

    for i in range(num_iterations):
        with tqdm(total=episodes_per_iteration, desc='Iteration %d' % i) as pbar:
            for i_episode in range(episodes_per_iteration):
                for agent in agents.values():
                    agent.reset_buffer()

                # Reset environment
                if global_episode == 0:
                    obs, infos = env.reset(options={'randomize': False})
                else:
                    obs, infos = env.reset(options={'randomize': randomize})

                episode_returns = {aid: 0.0 for aid in agents.keys()}
                episode_true_returns = {aid: 0.0 for aid in agents.keys()}
                done = False
                step = 0

                # --- Asynchronous state trackers ---
                # Which agents need to make a decision at the current timestep
                active_durations = {aid: 0 for aid in agents.keys()}
                
                # The action currently being executed by each agent
                current_actions = {aid: None for aid in agents.keys()}
                absolute_actions = {aid: None for aid in agents.keys()}
                
                # Storage for the start of an agent's macro-transition
                start_obs = {aid: None for aid in agents.keys()}
                start_global_state = {aid: None for aid in agents.keys()}
                
                # Reward accumulators for macro-transitions
                cumul_rewards = {aid: 0.0 for aid in agents.keys()}
                cumul_true_rewards = {aid: 0.0 for aid in agents.keys()}
                
                # Time elapsed within the current macro-transition for discounting
                time_in_macro = {aid: 0 for aid in agents.keys()}

                while not done:
                    # --- Build Global State for current timestep ---
                    global_state = np.concatenate([obs[aid] for aid in agent_keys], axis=0)

                    # --- Step 1: Decision point for agents whose duration has expired ---
                    for agent_id, agent in agents.items():
                        if active_durations[agent_id] <= 0:
                            # If they just finished a macro-transition, store it!
                            if start_obs[agent_id] is not None:
                                agent.store_transition(
                                    state=start_obs[agent_id],
                                    global_state=start_global_state[agent_id],
                                    action=current_actions[agent_id],
                                    next_global_state=global_state,
                                    reward=cumul_rewards[agent_id],
                                    done=False, # Wait till end of episode for terminal done
                                    duration=time_in_macro[agent_id],
                                    true_reward=cumul_true_rewards[agent_id],
                                )
                                episode_returns[agent_id] += cumul_rewards[agent_id]
                                episode_true_returns[agent_id] += cumul_true_rewards[agent_id]
                                
                                # Reset macro-accumulators
                                cumul_rewards[agent_id] = 0.0
                                cumul_true_rewards[agent_id] = 0.0
                                time_in_macro[agent_id] = 0

                            # Start a NEW macro-transition
                            agent_state = obs[agent_id]
                            action, duration, mu, sigma, dur_probs = agent.take_action(
                                agent_state, return_distribution=True
                            )
                            batch_policy_mu[agent_id].append(np.atleast_1d(mu))
                            batch_policy_sigma[agent_id].append(np.atleast_1d(sigma))
                            batch_duration_probs[agent_id].append(dur_probs)
                            batch_sampled_durations[agent_id].append(duration)

                            if delta_actions:
                                from rl.rl_utils import extract_current_gate_widths
                                current_gates = extract_current_gate_widths(obs[agent_id], agents[agent_id].act_dim)
                                absolute_action = current_gates + action
                                absolute_action = np.clip(
                                    absolute_action,
                                    agents[agent_id].act_low,
                                    agents[agent_id].act_high
                                )
                                absolute_actions[agent_id] = absolute_action
                            else:
                                absolute_actions[agent_id] = action

                            current_actions[agent_id] = action
                            active_durations[agent_id] = duration
                            
                            # Cache states for TD target
                            start_obs[agent_id] = obs[agent_id].copy()
                            start_global_state[agent_id] = global_state.copy()

                    # --- Step 2: Execute 1 step in the environment ---
                    next_obs, rewards, terms, truncs, infos = env.step(absolute_actions)
                    
                    # MAPPO: centralized critic trained on global reward 
                    global_reward_step = sum(rewards.values())
                    global_true_reward_step = sum(infos[aid].get('true_reward', rewards[aid]) for aid in rewards.keys())

                    for aid in agents.keys():
                        gamma = agents[aid].gamma
                        # Sum global rewards with temporal discounting relative to start of macro-transition
                        k_step = time_in_macro[aid]
                        cumul_rewards[aid] += global_reward_step * (gamma ** k_step)
                        cumul_true_rewards[aid] += global_true_reward_step
                        
                        # Tick duration
                        active_durations[aid] -= 1
                        time_in_macro[aid] += 1

                    obs = next_obs
                    step += 1
                    done = any(terms.values()) or any(truncs.values())
                    
                    # --- Step 3: Handle terminal state ---
                    if done:
                        final_global_state = np.concatenate([obs[aid] for aid in agent_keys], axis=0)
                        for agent_id, agent in agents.items():
                            if start_obs[agent_id] is not None:
                                agent.store_transition(
                                    state=start_obs[agent_id],
                                    global_state=start_global_state[agent_id],
                                    action=current_actions[agent_id],
                                    next_global_state=final_global_state,
                                    reward=cumul_rewards[agent_id],
                                    done=done,
                                    duration=time_in_macro[agent_id],
                                    true_reward=cumul_true_rewards[agent_id],
                                )
                                episode_returns[agent_id] += cumul_rewards[agent_id]
                                episode_true_returns[agent_id] += cumul_true_rewards[agent_id]
                        break

                # End of episode
                for agent_id, agent in agents.items():
                    agent.store_trajectory()
                    return_dict[agent_id].append(episode_returns[agent_id])
                    batch_returns[agent_id].append(episode_returns[agent_id])
                    batch_true_returns[agent_id].append(episode_true_returns[agent_id])

                global_episode += 1

                # Debug saves
                if debug_save_dir and debug_episodes and global_episode in debug_episodes:
                    run_idx = debug_episodes.index(global_episode) + 1
                    save_path = f"{debug_save_dir}_run{run_idx}"
                    env.save(save_path)
                    print(f"[Debug] Saved simulation at episode {global_episode} to {save_path}")

                # Check batch update
                first_agent = next(iter(agents.values()))
                if first_agent.get_batch_size() >= num_trajectories_per_update:
                    for agent_id, agent in agents.items():
                        if hasattr(env, 'ret_rms') and env.ret_rms is not None:
                            try:
                                agent.set_reward_normalizer_var(float(env.ret_rms.var))
                            except:
                                pass
                        agent.update_batch()

                    global_update += 1

                    # WandB logging
                    if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
                        log_dict = {
                            'update': global_update,
                            'episode': global_episode,
                            'batch_avg_normalized_return': np.mean(
                                [np.mean(batch_returns[aid]) for aid in agents.keys()]),
                            'batch_avg_true_return': np.mean(
                                [np.mean(batch_true_returns[aid]) for aid in agents.keys()]),
                            'trajectories_per_update': num_trajectories_per_update,
                            'episode_steps': step,
                        }
                        for agent_id in agents.keys():
                            log_dict[f'agent_{agent_id}_batch_avg_return'] = np.mean(
                                batch_returns[agent_id])
                            log_dict[f'agent_{agent_id}_batch_avg_true_return'] = np.mean(
                                batch_true_returns[agent_id])
                        # LR + entropy
                        first_agent_lr = first_agent.get_current_lr()
                        log_dict['actor_lr'] = first_agent_lr['actor_lr']
                        log_dict['critic_lr'] = first_agent_lr['critic_lr']
                        log_dict['entropy_coef'] = first_agent.entropy_coef
                        # Policy stats
                        for agent_id in agents.keys():
                            if batch_policy_mu[agent_id]:
                                mu_arr = np.array(batch_policy_mu[agent_id])
                                sigma_arr = np.array(batch_policy_sigma[agent_id])
                                avg_mu = np.mean(mu_arr, axis=0)
                                avg_sigma = np.mean(sigma_arr, axis=0)
                                for d in range(len(avg_mu)):
                                    log_dict[f'agent_{agent_id}_policy_mu_{d}'] = float(avg_mu[d])
                                    log_dict[f'agent_{agent_id}_policy_sigma_{d}'] = float(avg_sigma[d])
                            # Duration distribution (policy probabilities)
                            if batch_duration_probs[agent_id]:
                                dur_arr = np.array(batch_duration_probs[agent_id])
                                avg_dur = np.mean(dur_arr, axis=0)
                                for d_idx in range(len(avg_dur)):
                                    log_dict[f'agent_{agent_id}_dur_prob_{d_idx+1}'] = float(avg_dur[d_idx])
                            # Mean actually chosen duration (sampled integers)
                            if batch_sampled_durations[agent_id]:
                                log_dict[f'agent_{agent_id}_avg_chosen_duration'] = float(
                                    np.mean(batch_sampled_durations[agent_id]))
                        wandb.log(log_dict)

                    # Validation
                    if (agents_saved_dir
                            and global_update > (num_episodes // num_trajectories_per_update) // 2
                            and global_update % val_freq == 0):
                        best_avg_return = validate_and_save_best(
                            env, agents, agents_saved_dir,
                            delta_actions=delta_actions,
                            num_val_episodes=num_val_episodes,
                            randomize=True,
                            best_avg_return=best_avg_return,
                            global_episode=global_episode,
                            use_wandb=use_wandb and WANDB_AVAILABLE,
                        )

                    # Reset batch tracking
                    batch_returns = {aid: [] for aid in agents.keys()}
                    batch_true_returns = {aid: [] for aid in agents.keys()}
                    batch_policy_mu = {aid: [] for aid in agents.keys()}
                    batch_policy_sigma = {aid: [] for aid in agents.keys()}
                    batch_duration_probs = {aid: [] for aid in agents.keys()}
                    batch_sampled_durations = {aid: [] for aid in agents.keys()}

                # Progress bar
                if (i_episode + 1) % 10 == 0:
                    avg_return = np.mean([np.mean(return_dict[aid][-10:]) for aid in agents.keys()])
                    avg_true_return = np.mean(list(episode_true_returns.values()))
                    pbar.set_postfix({
                        'episode': '%d' % global_episode,
                        'update': '%d' % global_update,
                        'norm_ret': '%.3f' % avg_return,
                        'true_ret': '%.3f' % avg_true_return,
                        'steps': step
                    })
                pbar.update(1)

                for agent_id in agents.keys():
                    print(f"Agent {agent_id} episode reward: {episode_returns[agent_id].item():.3f}")
                print(f"All agents episode reward: {sum(episode_returns.values()).item():.3f}")

    final_returns = {aid: return_dict[aid][-1] if return_dict[aid] else 0.0
                     for aid in agents.keys()}
    return return_dict, final_returns



if __name__ == "__main__":
    algo = "mappo_hrl"
    SEED = 77
    NORM = False   
    builder_norm_obs = False  
    STATE_OPTION = "option3"
    randomize = True
    norm_ret = True
    action_gap = 1
    USE_PRETRAINED = False

    # set torch seed
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    # Enable deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dataset = "butterfly_scC"
    print("=" * 60)
    print(f"Fine-tuning {algo} Agents on PedNet Environment ({dataset})")
    print("=" * 60)

    # Create environment with normalization wrapper
    base_env = PedNetParallelEnv(
        dataset=dataset, normalize_obs=builder_norm_obs, obs_mode=STATE_OPTION, render_mode="animate", action_gap=action_gap
    )
    env = RunningNormalizeWrapper(base_env, norm_obs=NORM, norm_reward=norm_ret)
    env.seed(SEED)

    agents = {}
    agent_keys = sorted(list(env.possible_agents))
    global_obs_dim = sum([env.observation_space(aid).shape[0] for aid in agent_keys])
    
    for agent_id in agent_keys:
        agents[agent_id] = MAPPOAgentHRL(
            obs_dim=env.observation_space(agent_id).shape[0],
            global_obs_dim=global_obs_dim,
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
            num_episodes=400,
            tm_window=20,
            max_duration=7,
            duration_entropy_coef=0.05,
            duration_entropy_coef_min=0.001,
            value_fusion='mean',
        )

    if USE_PRETRAINED:
        # Load pretrained checkpoints (optional, but good for fine-tuning)
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
            # NOTE: For MAPPO, the critic is a different size. We only load the Actor weights.
            for agent_id, agent in agents.items():
                agent.actor.load_state_dict(source_agent_state['actor_state_dict'])
                # Cannot directly load value_net because global_obs_dim != obs_dim
                # agent.value_net.load_state_dict(source_agent_state['critic_state_dict'])
                print(f"Loaded pretrained actor weights for agent: {agent_id}")
        else:
            print("Warning: Could not find valid pretrained weights in checkpoint.")

    # Train MAPPO HRL agents
    print("Starting MAPPO Training Loop...")
    return_dict, _ = train_mappo_batch(
        env, agents, num_episodes=400, num_trajectories_per_update=2, delta_actions=True,
        randomize=randomize, agents_saved_dir=project_root / f"rl/checkpoints/mappo_hrl_{dataset}",
        num_val_episodes=10, val_freq=10, use_wandb=True,
        debug_save_dir=f"rl_training/{dataset}/mappo_hrl_debug",
        debug_save_episodes=[5, 50, 100, 200, 400]
    )
    # Evaluation phase
    SEED = 42
    randomized = True
    num_runs = 15

    # Load the best fine-tuned agents
    agents, config_data = load_all_agents(save_dir=project_root / f"rl/checkpoints/mappo_hrl_{dataset}", device="cpu")

    base_env = PedNetParallelEnv(
        dataset=dataset, normalize_obs=builder_norm_obs, obs_mode=STATE_OPTION, render_mode="animate"
    )
    env = RunningNormalizeWrapper(base_env, norm_obs=NORM, norm_reward=False, training=False)

    if config_data is not None and 'normalization_stats' in config_data:
        env.set_normalization_stats(config_data['normalization_stats'])

    rl_results = evaluate_agents(
        env, agents,
        delta_actions=True,
        deterministic=True,
        seed=SEED,
        randomize=randomized,
        num_runs=num_runs,
        save_dir=project_root / f"outputs/rl_training/{dataset}/{algo}"
    )

    # Compare with no control
    env = PedNetParallelEnv(
        dataset=dataset, normalize_obs=False, obs_mode="option2", render_mode="animate", action_gap=action_gap
    )
    no_control_agents = {agent_id: None for agent_id in env.possible_agents}
    no_control_results = evaluate_agents(
        env, no_control_agents,
        delta_actions=False,
        seed=SEED,
        no_control=True,
        randomize=randomized,
        num_runs=num_runs,
        save_dir=project_root / f"outputs/rl_training/{dataset}/no_control"
    )

    print("\n" + "=" * 60)
    print("Comparison of All Methods")
    print("=" * 60)
    print(f"Fine-tuned {algo} avg reward: {rl_results['avg_reward']:.3f}")
    print(f"No control avg reward:        {no_control_results['avg_reward']:.3f}")
    print("=" * 60)

    # render the results
    project_root = Path(__file__).resolve().parent.parent
    # Render final simulation
    env.render(
        simulation_dir=str(project_root / f"outputs/rl_training/{dataset}/{algo}_run5"),
        # simulation_dir=str(project_root / f"outputs/rl_training/{dataset}/no_control"),
        variable='density',
        vis_actions=True,
        save_dir=None
    )
