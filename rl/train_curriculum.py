# -*- coding: utf-8 -*-
"""
Train PPO-HRL agent using mixed-batch curriculum learning across multiple scenarios.

Instead of sequential curriculum (train on scA → scB → scC → scD), this script
mixes episodes from ALL scenarios within each training batch.  For example, with
num_trajectories_per_update=10 and 5 scenarios, each batch may contain ~2 episodes
from each scenario.  This yields better generalisation and avoids catastrophic
forgetting by construction.

Key design: a SINGLE agent is trained across all scenarios.  Different scenarios
may expose different agent IDs (e.g. gate_2 in scA vs gate_6 in scE), but the
underlying neural network is the same — we just dynamically look up the agent_id
from each environment.

Noise reduction strategies:
    1. Per-scenario reward rescaling: rewards in the batch buffer are rescaled so
       that each scenario has comparable return variance before the PPO update.
       This prevents scenarios with large reward magnitudes from dominating
       the advantage signal.
    2. Larger batch size: with ≥2 episodes per scenario per batch, the gradient
       is smoother and more representative.

Usage:
    python rl/train_curriculum.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import numpy as np
import torch
from collections import deque
from tqdm import tqdm
from rl import PedNetParallelEnv
from rl.rl_utils import (
    RunningNormalizeWrapper,
    save_all_agents,
    load_all_agents,
    evaluate_agents,
    validate_agents,
)
from rl.agents.PPO_hrl import PPOAgentHRL

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# Environment helpers
# =============================================================================

def create_env(dataset: str, obs_mode: str = "option3",
               normalize_obs: bool = False, norm_reward: bool = True,
               action_gap: int = 1) -> RunningNormalizeWrapper:
    """Create a RunningNormalizeWrapper-wrapped environment for a given scenario."""
    base_env = PedNetParallelEnv(
        dataset=dataset,
        normalize_obs=normalize_obs,
        obs_mode=obs_mode,
        render_mode="animate",
        action_gap=action_gap,
    )
    env = RunningNormalizeWrapper(base_env, norm_obs=False, norm_reward=norm_reward)
    return env


# =============================================================================
# Per-scenario reward scaler
# =============================================================================

class ScenarioRewardScaler:
    """
    Track per-scenario return statistics and rescale rewards so all scenarios
    have comparable return variance before the PPO update.

    The scaler maintains a rolling window of episode returns per scenario
    and computes a multiplicative factor to bring each scenario's reward std
    to a common target.  During a warmup phase (before enough episodes are
    collected), no scaling is applied.

    This is applied *after* the environment's RunningNormalizeWrapper and
    *before* the agent's update_batch().
    """

    def __init__(self, scenarios: list, target_std: float = 1.0,
                 warmup_episodes: int = 5, window_size: int = 50):
        """
        Args:
            scenarios: List of scenario names.
            target_std: Target standard deviation for episode returns.
            warmup_episodes: Minimum episodes per scenario before scaling.
            window_size: Rolling window size for return statistics.
        """
        self.target_std = target_std
        self.warmup = warmup_episodes
        self.returns = {sc: deque(maxlen=window_size) for sc in scenarios}

    def add_return(self, scenario: str, episode_return: float):
        """Record an episode return for a scenario."""
        self.returns[scenario].append(episode_return)

    def get_scale(self, scenario: str) -> float:
        """
        Return a multiplicative factor for rewards from the given scenario.

        Before warmup completes: returns 1.0 (no scaling).
        After warmup: returns target_std / scenario_std.
        """
        returns = self.returns[scenario]
        if len(returns) < self.warmup:
            return 1.0
        std = float(np.std(returns))
        if std < 1e-6:
            return 1.0
        return self.target_std / std

    def rescale_batch_buffer(self, batch_buffer: list, batch_scenarios: list):
        """
        In-place rescale the rewards in the agent's batch_buffer.

        Each trajectory in batch_buffer corresponds to one episode, and
        batch_scenarios[i] records which scenario that episode came from.
        We rescale rewards[i] *= scale(scenario_i).

        Args:
            batch_buffer: The agent's batch_buffer (list of trajectory dicts).
            batch_scenarios: Parallel list of scenario names, same length.
        """
        assert len(batch_buffer) == len(batch_scenarios), (
            f"batch_buffer ({len(batch_buffer)}) and batch_scenarios "
            f"({len(batch_scenarios)}) length mismatch"
        )
        for traj, sc in zip(batch_buffer, batch_scenarios):
            scale = self.get_scale(sc)
            if abs(scale - 1.0) > 1e-6:
                traj['rewards'] = traj['rewards'] * scale

    def stats_summary(self) -> dict:
        """Return per-scenario stats for logging."""
        summary = {}
        for sc, rets in self.returns.items():
            if len(rets) > 0:
                summary[sc] = {
                    'n': len(rets),
                    'mean': float(np.mean(rets)),
                    'std': float(np.std(rets)),
                    'scale': self.get_scale(sc),
                }
        return summary


# =============================================================================
# Mixed-batch curriculum training loop (HRL, single agent)
# =============================================================================

def train_hrl_curriculum_mixed(
    agent: PPOAgentHRL,
    scenario_sequence: list,
    env_factory,
    num_episodes: int = 400,
    num_trajectories_per_update: int = 10,
    delta_actions: bool = True,
    randomize: bool = True,
    val_freq: int = 10,
    num_val_episodes: int = 10,
    save_dir: str = None,
    use_wandb: bool = True,
    scenario_sampling: str = "round_robin",
    debug_save_dir: str = None,
    debug_save_episodes: list = None,
    reward_rescaling: bool = True,
    rescale_warmup: int = 5,
):
    """
    Train a SINGLE HRL agent with mixed-scenario batches (curriculum learning).

    At each episode the environment is drawn from the scenario pool (round-robin
    or random).  The agent_id is obtained dynamically from each scenario's
    environment, so different scenarios can have different controller IDs.
    Trajectories from different scenarios are mixed inside the same batch buffer,
    so every PPO update sees data from multiple demand patterns.

    Noise reduction features:
        - Per-scenario reward rescaling (Strategy 1): equalises reward variance
          across scenarios before update_batch(), preventing any single scenario
          from dominating the advantage signal.
        - Larger batch size (Strategy 3): more trajectories per update gives
          smoother gradients.  Recommended: ≥ 2 episodes per scenario per batch.

    Args:
        agent:  A single PPOAgentHRL instance.
        scenario_sequence:  List of dataset names.
        env_factory:  Callable(dataset: str) -> env
        num_episodes:  Total training episodes (across all scenarios combined)
        num_trajectories_per_update:  Episodes per PPO update batch.
                 Recommended: 2× len(scenario_sequence) for balanced batches.
        delta_actions:  Whether agents use delta actions
        randomize:  Whether to randomize env at reset
        val_freq:  Validate every N updates
        num_val_episodes:  Episodes per validation run
        save_dir:  Checkpoint directory (best model saved here)
        use_wandb:  Log to WandB
        scenario_sampling:  "round_robin" or "random"
        debug_save_dir:  Algorithm label for debug saves
        debug_save_episodes:  Episode numbers to save debug simulations
        reward_rescaling:  If True, rescale rewards per-scenario before updates
        rescale_warmup:  Episodes per scenario before rescaling kicks in

    Returns:
        return_list:  [episode_returns]
        final_return:  float
    """
    if use_wandb and WANDB_AVAILABLE:
        if wandb.run is None:
            wandb.init(
                project="crowd-control-rl",
                name=f"ppo-hrl-curriculum-{'_'.join(s.split('_')[-1] for s in scenario_sequence)}",
            )

    # --- Cache environments (one per scenario) ---
    envs = {}
    scenario_agent_ids = {}
    for sc in scenario_sequence:
        envs[sc] = env_factory(sc)
        scenario_agent_ids[sc] = envs[sc].possible_agents[0]
    print(f"Cached {len(envs)} scenario environments:")
    for sc, aid in scenario_agent_ids.items():
        print(f"  {sc} -> agent_id: {aid}")

    # --- Per-scenario reward scaler ---
    reward_scaler = ScenarioRewardScaler(
        scenario_sequence, target_std=1.0, warmup_episodes=rescale_warmup
    )

    # --- Initialise tracking ---
    return_list = []
    global_episode = 0
    global_update = 0
    best_avg_return = float('-inf')

    debug_episodes = tuple(debug_save_episodes) if debug_save_episodes else None

    agent.init_batch_buffer()

    # Track which scenario each trajectory in the current batch came from
    batch_scenario_tags = []

    # Adjust total_updates for entropy / noise decay
    if hasattr(agent, "total_updates"):
        effective_updates = max(
            1,
            int(num_episodes / float(max(1, num_trajectories_per_update)) * 0.8),
        )
        agent.total_updates = effective_updates

    # Batch-level tracking (reset after each update)
    def _reset_batch_tracking():
        return {
            'returns': [],
            'true_returns': [],
            'policy_mu': [],
            'policy_sigma': [],
            'duration_probs': [],
            'sampled_durations': [],
            'scenarios': [],
        }

    batch_track = _reset_batch_tracking()

    # --- Iteration structure ---
    num_iterations = 10
    episodes_per_iteration = num_episodes // num_iterations

    scenario_idx_counter = 0

    for i in range(num_iterations):
        with tqdm(total=episodes_per_iteration, desc='Iteration %d' % i) as pbar:
            for i_episode in range(episodes_per_iteration):
                # --- Pick scenario for this episode ---
                if scenario_sampling == "round_robin":
                    sc = scenario_sequence[scenario_idx_counter % len(scenario_sequence)]
                    scenario_idx_counter += 1
                else:
                    sc = np.random.choice(scenario_sequence)

                env = envs[sc]
                agent_id = scenario_agent_ids[sc]
                batch_track['scenarios'].append(sc)

                # Reset agent for new episode
                agent.reset_buffer()

                # Reset environment
                if global_episode == 0:
                    obs, infos = env.reset(options={'randomize': False})
                else:
                    obs, infos = env.reset(options={'randomize': randomize})

                episode_return = 0.0
                episode_true_return = 0.0
                done = False
                step = 0

                # --- Episode loop (HRL macro-steps) ---
                while not done:
                    agent_state = obs[agent_id]
                    action, duration, mu, sigma, dur_probs = agent.take_action(
                        agent_state, return_distribution=True
                    )
                    batch_track['policy_mu'].append(np.atleast_1d(mu))
                    batch_track['policy_sigma'].append(np.atleast_1d(sigma))
                    batch_track['duration_probs'].append(dur_probs)
                    batch_track['sampled_durations'].append(duration)

                    if delta_actions:
                        absolute_action = (
                            obs[agent_id].reshape(agent.act_dim, -1)[:, -1] + action
                        )
                        absolute_action = np.clip(
                            absolute_action, agent.act_low, agent.act_high
                        )
                    else:
                        absolute_action = action

                    # --- Execute for k steps (duration) ---
                    cumul_reward = 0.0
                    cumul_true_reward = 0.0
                    start_obs = obs[agent_id].copy()

                    for k_step in range(duration):
                        if done:
                            break  # simulation already ended in a previous step

                        next_obs, rewards, terms, truncs, infos = env.step(
                            {agent_id: absolute_action}
                        )

                        gamma = agent.gamma
                        cumul_reward += rewards[agent_id] * (gamma ** k_step)
                        if agent_id in infos and 'true_reward' in infos[agent_id]:
                            cumul_true_reward += infos[agent_id]['true_reward']
                        else:
                            cumul_true_reward += rewards[agent_id]

                        obs = next_obs
                        step += 1
                        done = any(terms.values()) or any(truncs.values())

                    # --- Store macro-transition ---
                    agent.store_transition(
                        state=start_obs,
                        action=action,
                        next_state=obs[agent_id],
                        reward=cumul_reward,
                        done=done,
                        duration=duration,
                        true_reward=cumul_true_reward,
                    )
                    episode_return += cumul_reward
                    episode_true_return += cumul_true_reward

                # --- End of episode ---
                agent.store_trajectory()
                batch_scenario_tags.append(sc)  # tag this trajectory
                reward_scaler.add_return(sc, episode_return)  # track stats

                return_list.append(episode_return)
                batch_track['returns'].append(episode_return)
                batch_track['true_returns'].append(episode_true_return)

                global_episode += 1

                # Debug saves
                if debug_save_dir and debug_episodes and global_episode in debug_episodes:
                    run_idx = debug_episodes.index(global_episode) + 1
                    save_path = f"rl_training/{sc}/{debug_save_dir}_run{run_idx}"
                    env.save(save_path)
                    print(f"[Debug] Saved simulation at episode {global_episode} ({sc}) "
                          f"to outputs/{save_path}\n"
                          f"  → visualize with: visualize_run('{sc}', '{debug_save_dir}', run_id={run_idx})")

                # --- Batch update ---
                if agent.get_batch_size() >= num_trajectories_per_update:
                    # Strategy 1: Per-scenario reward rescaling
                    # Rescale rewards in the batch buffer so all scenarios have
                    # comparable return variance before advantage computation.
                    if reward_rescaling:
                        reward_scaler.rescale_batch_buffer(
                            agent.batch_buffer, batch_scenario_tags
                        )

                    if hasattr(env, 'ret_rms') and env.ret_rms is not None:
                        try:
                            agent.set_reward_normalizer_var(float(env.ret_rms.var))
                        except Exception:
                            pass
                    agent.update_batch()

                    global_update += 1
                    batch_scenario_tags = []  # reset tags for next batch

                    # --- WandB logging ---
                    if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
                        log_dict = {
                            'update': global_update,
                            'episode': global_episode,
                            'batch_avg_normalized_return': np.mean(batch_track['returns']),
                            'batch_avg_true_return': np.mean(batch_track['true_returns']),
                            'trajectories_per_update': num_trajectories_per_update,
                            'episode_steps': step,
                            'batch_scenarios': ', '.join(batch_track['scenarios']),
                        }

                        # LR + entropy
                        agent_lr = agent.get_current_lr()
                        log_dict['actor_lr'] = agent_lr['actor_lr']
                        log_dict['critic_lr'] = agent_lr['critic_lr']
                        log_dict['entropy_coef'] = agent.entropy_coef

                        # Policy stats
                        if batch_track['policy_mu']:
                            mu_arr = np.array(batch_track['policy_mu'])
                            sigma_arr = np.array(batch_track['policy_sigma'])
                            avg_mu = np.mean(mu_arr, axis=0)
                            avg_sigma = np.mean(sigma_arr, axis=0)
                            for d in range(len(avg_mu)):
                                log_dict[f'policy_mu_{d}'] = float(avg_mu[d])
                                log_dict[f'policy_sigma_{d}'] = float(avg_sigma[d])
                        if batch_track['duration_probs']:
                            dur_arr = np.array(batch_track['duration_probs'])
                            avg_dur = np.mean(dur_arr, axis=0)
                            for d_idx in range(len(avg_dur)):
                                log_dict[f'dur_prob_{d_idx+1}'] = float(avg_dur[d_idx])
                        if batch_track['sampled_durations']:
                            log_dict['avg_chosen_duration'] = float(
                                np.mean(batch_track['sampled_durations']))

                        # Per-scenario reward scaling stats
                        if reward_rescaling:
                            scaler_stats = reward_scaler.stats_summary()
                            for sc_name, sc_stats in scaler_stats.items():
                                short = sc_name.split('_')[-1]
                                log_dict[f'reward_scale_{short}'] = sc_stats['scale']
                                log_dict[f'return_mean_{short}'] = sc_stats['mean']
                                log_dict[f'return_std_{short}'] = sc_stats['std']

                        wandb.log(log_dict)

                    # --- Validation (on ALL scenarios) ---
                    if (save_dir
                            and global_update > (num_episodes // num_trajectories_per_update) // 5
                            and global_update % val_freq == 0):
                        val_returns = []
                        for val_sc in scenario_sequence:
                            val_env = envs[val_sc]
                            val_aid = scenario_agent_ids[val_sc]
                            val_result = validate_agents(
                                val_env, {val_aid: agent},
                                delta_actions=delta_actions,
                                num_episodes=num_val_episodes,
                                randomize=True,
                            )
                            val_returns.append(val_result['avg_return'])

                            if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
                                wandb.log({
                                    f'val_{val_sc}_return': val_result['avg_return'],
                                    'val_update': global_update,
                                })

                        avg_val_return = np.mean(val_returns)
                        if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
                            wandb.log({
                                'val_avg_all_scenarios': avg_val_return,
                                'val_update': global_update,
                            })

                        if avg_val_return > best_avg_return:
                            best_avg_return = avg_val_return
                            canonical_aid = scenario_agent_ids[scenario_sequence[0]]
                            save_all_agents(
                                {canonical_aid: agent}, save_dir,
                                metadata={
                                    'episode': int(global_episode),
                                    'update': int(global_update),
                                    'val_avg_all_scenarios': float(avg_val_return),
                                    'val_per_scenario': {
                                        s: float(r) for s, r in zip(scenario_sequence, val_returns)
                                    },
                                    'scenarios': scenario_sequence,
                                    'scenario_agent_ids': scenario_agent_ids,
                                },
                            )
                            print(f"\n[Val] New best avg return across all scenarios: "
                                  f"{best_avg_return:.3f} at update {global_update} "
                                  f"(per-scenario: {dict(zip(scenario_sequence, [f'{r:.3f}' for r in val_returns]))})")

                    # Reset batch tracking
                    batch_track = _reset_batch_tracking()

                # --- Progress bar ---
                if (i_episode + 1) % 10 == 0:
                    avg_return = np.mean(return_list[-10:]) if return_list else 0.0
                    pbar.set_postfix({
                        'ep': '%d' % global_episode,
                        'upd': '%d' % global_update,
                        'sc': sc.split('_')[-1],
                        'norm_ret': '%.3f' % avg_return,
                        'true_ret': '%.3f' % episode_true_return,
                        'steps': step,
                    })
                pbar.update(1)

                r_val = episode_return.item() if hasattr(episode_return, 'item') else episode_return
                print(f"Agent [{sc.split('_')[-1]}, {agent_id}] episode reward: {r_val:.3f}")

    final_return = return_list[-1] if return_list else 0.0
    return return_list, final_return


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # ======================================================================
    # Configuration
    # ======================================================================
    algo = "ppo_hrl"
    SEED = 77
    NORM = False
    builder_norm_obs = False
    STATE_OPTION = "option4" # only option 3 or option 4. option 3 is flow, option 4 include density
    randomize = True
    norm_ret = True
    action_gap = 1

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Curriculum settings
    SCENARIO_SEQUENCE = [
        "butterfly_scB",
        # "butterfly_scA",
        "butterfly_scC",
        "butterfly_scD",
        "butterfly_scE",
    ]
    NUM_EPISODES = 600           # total episodes across all scenarios
    NUM_TRAJ_PER_UPDATE = 10     # ≥ 2 per scenario for balanced batches
    VAL_FREQ = 10                 # validate every N updates
    NUM_VAL_EPISODES = 5
    SCENARIO_SAMPLING = "round_robin"  # or "random"

    print("=" * 60)
    print(f"Mixed-Batch Curriculum Training — {algo.upper()}")
    print(f"Scenarios: {SCENARIO_SEQUENCE}")
    print(f"Total episodes: {NUM_EPISODES}")
    print(f"Trajectories per update: {NUM_TRAJ_PER_UPDATE}")
    print(f"Sampling: {SCENARIO_SAMPLING}")
    print("=" * 60)

    # ======================================================================
    # Environment factory
    # ======================================================================
    def env_factory(dataset: str):
        return create_env(
            dataset,
            obs_mode=STATE_OPTION,
            normalize_obs=builder_norm_obs,
            norm_reward=norm_ret,
            action_gap=action_gap,
        )

    # ======================================================================
    # Create SINGLE agent (use first scenario to obtain obs/act dims)
    # ======================================================================
    ref_env = env_factory(SCENARIO_SEQUENCE[0])
    ref_env.seed(SEED)
    ref_agent_id = ref_env.possible_agents[0]
    print(f"Reference agent from {SCENARIO_SEQUENCE[0]}: {ref_agent_id}")

    agent = PPOAgentHRL(
        obs_dim=ref_env.observation_space(ref_agent_id).shape[0],
        act_dim=ref_env.action_space(ref_agent_id).shape[0],
        act_low=ref_env.action_space(ref_agent_id).low,
        act_high=ref_env.action_space(ref_agent_id).high,
        actor_lr=1e-4,
        critic_lr=2e-4,
        use_lr_decay=False,
        gamma=0.99,
        lmbda=0.96,
        entropy_coef=0.07,
        kl_tolerance=0.02,
        use_delta_actions=True,
        max_delta=2.5,
        lstm_hidden_size=64,
        num_lstm_layers=2,
        num_heads=2,
        use_param_noise=False,
        use_action_noise=False,
        num_episodes=NUM_EPISODES,
        tm_window=50,
        max_duration=7,
        duration_entropy_coef=0.05,
        duration_entropy_coef_min=0.005,
        value_fusion='mean',
    )
    del ref_env

    save_dir = f"./checkpoints/curriculum_{algo}"

    # ======================================================================
    # Train
    # ======================================================================
    return_list, final_return = train_hrl_curriculum_mixed(
        agent=agent,
        scenario_sequence=SCENARIO_SEQUENCE,
        env_factory=env_factory,
        num_episodes=NUM_EPISODES,
        num_trajectories_per_update=NUM_TRAJ_PER_UPDATE,
        delta_actions=True,
        randomize=randomize,
        val_freq=VAL_FREQ,
        num_val_episodes=NUM_VAL_EPISODES,
        save_dir=save_dir,
        use_wandb=True,
        scenario_sampling=SCENARIO_SAMPLING,
        debug_save_dir="curriculum_debug",
        debug_save_episodes=[10, 100, 200, 300, 400],
        reward_rescaling=True,
        rescale_warmup=5,
    )

    # ======================================================================
    # Extended Evaluation (load best model)
    # Saves simulation data so you can visualize with:
    #   visualize_run(dataset=sc, algorithm="curriculum_eval", run_id=N)
    # or evaluate metrics with:
    #   evaluate_all_runs(base_dir="outputs/rl_training", dataset=sc,
    #                     algorithms=["curriculum_eval"])
    # ======================================================================
    print("\n" + "=" * 60)
    print("Extended Evaluation (10 runs per scenario)")
    print("=" * 60)

    EVAL_SEED = 42
    NUM_EVAL_RUNS = 5
    EVAL_ALGO_LABEL = "curriculum_eval"  # label for saved outputs

    # Evaluate on ALL scenarios (including ones not trained on, to test generalisation)
    EVAL_SCENARIOS = [
        "butterfly_scA",
        "butterfly_scB",
        "butterfly_scC",
        "butterfly_scD",
        "butterfly_scE",
    ]

    loaded_agents, config_data = load_all_agents(save_dir=save_dir, device="cpu")
    loaded_agent = list(loaded_agents.values())[0]

    for sc in EVAL_SCENARIOS:
        eval_env = env_factory(sc)
        eval_agent_id = eval_env.possible_agents[0]
        # Save dir follows visualize_run convention:
        #   outputs/rl_training/{dataset}/{algorithm}_run{N}/
        eval_save_dir = f"rl_training/{sc}/{EVAL_ALGO_LABEL}"
        eval_results = evaluate_agents(
            eval_env, {eval_agent_id: loaded_agent},
            delta_actions=True,
            deterministic=True,
            seed=EVAL_SEED,
            randomize=True,
            num_runs=NUM_EVAL_RUNS,
            save_dir=eval_save_dir,
            verbose=False,
        )
        print(f"  {sc}: {eval_results['avg_reward']:.3f} ± {eval_results['avg_reward_std']:.3f}")
        print(f"    → saved {NUM_EVAL_RUNS} runs to outputs/{eval_save_dir}_run*/")
        print(f"    → visualize: visualize_run('{sc}', '{EVAL_ALGO_LABEL}', run_id=1)")

    print("=" * 60)
def train_hrl_curriculum_incremental(
    agent,
    scenario_sequence: list,
    env_factory,
    total_episodes: int = 1000,
    num_trajectories_per_update: int = 10,
    delta_actions: bool = True,
    randomize: bool = True,
    val_freq: int = 10,
    num_val_episodes: int = 5,
    save_dir: str = None,
    use_wandb: bool = True,
    reward_rescaling: bool = True,
    rescale_warmup: int = 5,
    threshold_return: float = -500.0,
    min_episodes_per_stage: int = 100,
):
    """
    Incremental Mix-In Curriculum.
    Starts by training ONLY on scenario 1. Once validation score on scenario 1
    exceeds threshold, expands the pool to [scenario 1, scenario 2]. And so on,
    until all scenarios are mixed in.
    """
    import numpy as np
    from tqdm import tqdm
    from rl.rl_utils import save_all_agents, validate_agents
    from rl.train_curriculum import ScenarioRewardScaler
    try:
        import wandb
        WANDB_AVAILABLE = True
    except ImportError:
        WANDB_AVAILABLE = False
        
    if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({"curriculum_type": "incremental_mixin"})
        
    # --- Cache environments (one per scenario) ---
    envs = {}
    scenario_agent_ids = {}
    for sc in scenario_sequence:
        envs[sc] = env_factory(sc)
        scenario_agent_ids[sc] = envs[sc].possible_agents[0]
        
    print(f"Cached {len(envs)} scenario environments:")
    for sc, aid in scenario_agent_ids.items():
        print(f"  {sc} -> agent_id: {aid}")
        
    # Per-scenario reward scaler
    reward_scaler = ScenarioRewardScaler(
        scenario_sequence, target_std=1.0, warmup_episodes=rescale_warmup
    )
    
    # State tracking
    stage_idx = 0
    active_scenarios = [scenario_sequence[stage_idx]]
    episodes_in_current_stage = 0
    
    global_episode = 0
    global_update = 0
    best_avg_return = float('-inf')
    all_returns = []
    
    print(f"\n{'=' * 60}")
    print(f"Incremental Stage {stage_idx + 1}/{len(scenario_sequence)}")
    print(f"Active scenarios: {active_scenarios}")
    print(f"{'=' * 60}")
    
    agent.init_batch_buffer()
    batch_scenario_tags = []
    
    if hasattr(agent, "total_updates"):
        effective_updates = max(
            1,
            int(total_episodes / float(max(1, num_trajectories_per_update)) * 0.8),
        )
        agent.total_updates = effective_updates
        
    def _reset_batch_tracking():
        return {
            'returns': [],
            'true_returns': [],
            'policy_mu': [],
            'policy_sigma': [],
            'duration_probs': [],
            'sampled_durations': [],
            'scenarios': [],
        }
        
    batch_track = _reset_batch_tracking()
    
    scenario_idx_counter = 0

    with tqdm(total=total_episodes, desc="Training") as pbar:
        while global_episode < total_episodes:
            # --- Pick scenario from active pool (round robin) ---
            sc = active_scenarios[scenario_idx_counter % len(active_scenarios)]
            scenario_idx_counter += 1
            
            env = envs[sc]
            agent_id = scenario_agent_ids[sc]
            batch_track['scenarios'].append(sc)
            
            agent.reset_buffer()
            if global_episode == 0:
                obs, infos = env.reset(options={'randomize': False})
            else:
                obs, infos = env.reset(options={'randomize': randomize})
                
            episode_return = 0.0
            episode_true_return = 0.0
            done = False
            step = 0
            
            while not done:
                agent_state = obs[agent_id]
                action, duration, mu, sigma, dur_probs = agent.take_action(
                    agent_state, return_distribution=True
                )
                batch_track['policy_mu'].append(np.atleast_1d(mu))
                batch_track['policy_sigma'].append(np.atleast_1d(sigma))
                batch_track['duration_probs'].append(dur_probs)
                batch_track['sampled_durations'].append(duration)
                
                if delta_actions:
                    absolute_action = (
                        obs[agent_id].reshape(agent.act_dim, -1)[:, -1] + action
                    )
                    absolute_action = np.clip(
                        absolute_action, agent.act_low, agent.act_high
                    )
                else:
                    absolute_action = action
                    
                cumul_reward = 0.0
                cumul_true_reward = 0.0
                start_obs = obs[agent_id].copy()
                
                for k_step in range(duration):
                    if done:
                        break
                    
                    next_obs, rewards, terms, truncs, infos = env.step(
                        {agent_id: absolute_action}
                    )
                    
                    gamma = agent.gamma
                    cumul_reward += rewards[agent_id] * (gamma ** k_step)
                    if agent_id in infos and 'true_reward' in infos[agent_id]:
                        cumul_true_reward += infos[agent_id]['true_reward']
                    else:
                        cumul_true_reward += rewards[agent_id]
                        
                    obs = next_obs
                    step += 1
                    done = any(terms.values()) or any(truncs.values())
                    
                agent.store_transition(
                    state=start_obs,
                    action=action,
                    next_state=obs[agent_id],
                    reward=cumul_reward,
                    done=done,
                    duration=duration,
                    true_reward=cumul_true_reward,
                )
                episode_return += cumul_reward
                episode_true_return += cumul_true_reward
                
            agent.store_trajectory()
            batch_scenario_tags.append(sc)
            reward_scaler.add_return(sc, episode_return)
            
            all_returns.append(episode_return)
            batch_track['returns'].append(episode_return)
            batch_track['true_returns'].append(episode_true_return)
            
            global_episode += 1
            episodes_in_current_stage += 1
            pbar.update(1)
            
            # --- Batch Update ---
            if agent.get_batch_size() >= num_trajectories_per_update:
                if reward_rescaling:
                    reward_scaler.rescale_batch_buffer(
                        agent.batch_buffer, batch_scenario_tags
                    )
                    
                if hasattr(env, 'ret_rms') and env.ret_rms is not None:
                    try:
                        agent.set_reward_normalizer_var(float(env.ret_rms.var))
                    except Exception:
                        pass
                        
                agent.update_batch()
                global_update += 1
                batch_scenario_tags = []
                
                # --- WandB Logging ---
                if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
                    log_dict = {
                        'update': global_update,
                        'episode': global_episode,
                        'stage': stage_idx,
                        'batch_avg_normalized_return': np.mean(batch_track['returns']),
                        'batch_avg_true_return': np.mean(batch_track['true_returns']),
                        'batch_scenarios': ', '.join(batch_track['scenarios']),
                    }
                    if reward_rescaling:
                        scaler_stats = reward_scaler.stats_summary()
                        for sc_name, sc_stats in scaler_stats.items():
                            short = sc_name.split('_')[-1]
                            log_dict[f'reward_scale_{short}'] = sc_stats['scale']
                            log_dict[f'return_mean_{short}'] = sc_stats['mean']
                    wandb.log(log_dict)
                    
                # --- Validation & Curriculum Stage Advancement ---
                if global_update % val_freq == 0:
                    val_returns = []
                    # Validate on ALL scenarios, both active and strictly unseen
                    for val_sc in scenario_sequence:
                        val_env = envs[val_sc]
                        val_aid = scenario_agent_ids[val_sc]
                        val_result = validate_agents(
                            val_env, {val_aid: agent},
                            delta_actions=delta_actions,
                            num_episodes=num_val_episodes,
                            randomize=True,
                        )
                        val_returns.append(val_result['avg_return'])
                        
                        if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
                            wandb.log({
                                f'val_{val_sc}_return': val_result['avg_return'],
                                'val_update': global_update,
                            })
                            
                    avg_val_return = np.mean(val_returns)
                    if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
                        wandb.log({
                            'val_avg_all_scenarios': avg_val_return,
                            'val_update': global_update,
                        })
                        
                    # Save best generalist agent over the whole progression
                    if save_dir and avg_val_return > best_avg_return:
                        best_avg_return = avg_val_return
                        canonical_aid = scenario_agent_ids[scenario_sequence[0]]
                        save_all_agents(
                            {canonical_aid: agent}, save_dir,
                            metadata={
                                'episode': int(global_episode),
                                'update': int(global_update),
                                'stage': stage_idx,
                                'val_avg_all_scenarios': float(avg_val_return),
                            },
                        )

                    # --- stage advancement logic ---
                    # Check if we should advance to the next stage by measuring performance 
                    # specifically on the latest scenario added to the active pool
                    if stage_idx < len(scenario_sequence) - 1:
                        target_scenario = scenario_sequence[stage_idx]
                        target_scenario_val = val_returns[stage_idx]
                        
                        if (episodes_in_current_stage >= min_episodes_per_stage and 
                            target_scenario_val >= threshold_return):
                            stage_idx += 1
                            active_scenarios.append(scenario_sequence[stage_idx])
                            episodes_in_current_stage = 0
                            print(f"\n\n{'=' * 60}")
                            print(f"Target scenario '{target_scenario}' reached val threshold: {target_scenario_val:.2f} >= {threshold_return}")
                            print(f"Advancing to Stage {stage_idx + 1}/{len(scenario_sequence)}")
                            print(f"New Active Scenario Pool: {active_scenarios}")
                            print(f"{'=' * 60}\n")
                            if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
                                wandb.log({"curriculum_stage": stage_idx, "update": global_update})
                            
                            
                batch_track = _reset_batch_tracking()

    final_return = all_returns[-1] if all_returns else 0.0
    return all_returns, final_return
