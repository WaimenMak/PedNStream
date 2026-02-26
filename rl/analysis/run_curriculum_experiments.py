# -*- coding: utf-8 -*-
"""
Curriculum Training Experiment Runner.

Automates hyperparameter sweeps and curriculum-type comparisons for the
PPO-HRL agent across multiple butterfly scenarios.  This script does NOT
modify any existing project file — it imports from train_curriculum.py and
rl_utils.py.

Experiment axes
───────────────
1. Curriculum type: "mixed" (all scenarios in every batch) vs "sequential"
   (train on scenarios one-by-one, carrying weights forward).
2. Architecture: num_lstm_layers × num_heads × lstm_hidden_size.
3. State representation: option3 (flow only) vs option4 (flow + density).
4. Other tunables: actor_lr, entropy_coef, value_fusion, max_duration.

Results are logged to WandB (one run per config) and best checkpoints are
saved under  ./checkpoints/experiments/<experiment_tag>/

Usage
─────
    # Full sweep (all configs):
    python rl/run_curriculum_experiments.py

    # Smoke-test (5 episodes, no wandb):
    python rl/run_curriculum_experiments.py --test-run

    # Run a single config by index:
    python rl/run_curriculum_experiments.py --config-index 3

    # Sequential curriculum only:
    python rl/run_curriculum_experiments.py --curriculum sequential
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import argparse
import copy
import itertools
import json
import os
import time
from datetime import datetime

import numpy as np
import torch

# ── project imports (no modifications to these files) ──────────────────────
from rl.train_curriculum import (
    create_env,
    train_hrl_curriculum_mixed,
    train_hrl_curriculum_incremental,
    ScenarioRewardScaler,
)
from rl.rl_utils import (
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


# =========================================================================
# Scenario definitions
# =========================================================================

# Ordered by difficulty (approx.), used as curriculum stages for sequential.
ALL_SCENARIOS = [
    "butterfly_scB",  # 1→1, varied gate/link widths (easier routing task)
    "butterfly_scD",  # 2→2, OD flows, controller @ node 2 (harder routing task)
    "butterfly_scA",  # 1 Origin → 1 Dest, controller @ node 2 (inflow limiting)
    "butterfly_scC",  # 2 Origins → 1 Dest (intersecting demand, inflow limiting)
    "butterfly_scE",  # 2→2, OD flows, controller @ node 6 (shifted)
]

# Scenarios used for final evaluation (includes all + possibly unseen scF)
EVAL_SCENARIOS = [
    "butterfly_scA",
    "butterfly_scB",
    "butterfly_scC",
    "butterfly_scD",
    "butterfly_scE",
    "butterfly_scF",
]


# =========================================================================
# Hyperparameter grid
# =========================================================================

def build_experiment_grid(curriculum_filter: str = None):
    """
    Return a list of config dicts, one per experiment.

    Each dict has:
        curriculum_type: "mixed" | "sequential" | "incremental"
        num_lstm_layers, num_heads, lstm_hidden_size,
        state_option, actor_lr, critic_lr, entropy_coef,
        value_fusion, max_duration, gamma, lmbda,
        num_episodes, num_traj_per_update, ...
    """
    # ── axes to sweep ──────────────────────────────────────────────────────
    curriculum_types = ["mixed", "sequential", "incremental"]
    if curriculum_filter:
        curriculum_types = [curriculum_filter]

    architectures = [
        # (num_lstm_layers, num_heads, lstm_hidden_size)
        (1, 2, 64),   # baseline (current)
        (2, 2, 64),   # deeper LSTM
        (1, 4, 64),   # more attention heads
        (2, 4, 128),  # larger model
    ]

    state_options = ["option3", "option4"]  # option3=flow, option4=flow+density

    value_fusions = ["mean", "gated"]

    # ── fixed hyperparameters ──────────────────────────────────────────────
    fixed = dict(
        actor_lr=1e-4,
        critic_lr=2e-4,
        gamma=0.99,
        lmbda=0.96,
        entropy_coef=0.07,
        kl_tolerance=0.02,
        max_delta=2.5,
        max_duration=7,
        duration_entropy_coef=0.05,
        duration_entropy_coef_min=0.005,
        use_delta_actions=True,
        use_param_noise=False,
        use_action_noise=False,
        use_lr_decay=False,
        tm_window=50,
        num_episodes=600,
        num_traj_per_update=10,
        val_freq=10,
        num_val_episodes=5,
        scenario_sampling="round_robin",
        reward_rescaling=True,
        rescale_warmup=5,
        threshold_return=-500.0,
        min_episodes_per_stage=50,
        seed=77,
    )

    # ── build grid ─────────────────────────────────────────────────────────
    configs = []
    for cur_type in curriculum_types:
        for (n_lstm, n_heads, h_size) in architectures:
            for state_opt in state_options:
                for vf in value_fusions:
                    cfg = copy.deepcopy(fixed)
                    cfg["curriculum_type"] = cur_type
                    cfg["num_lstm_layers"] = n_lstm
                    cfg["num_heads"] = n_heads
                    cfg["lstm_hidden_size"] = h_size
                    cfg["state_option"] = state_opt
                    cfg["value_fusion"] = vf

                    # Tag for WandB and checkpoint dir
                    tag = (
                        f"{cur_type}_lstm{n_lstm}_h{n_heads}_hs{h_size}"
                        f"_{state_opt}_{vf}"
                    )
                    cfg["experiment_tag"] = tag
                    configs.append(cfg)

    return configs


# =========================================================================
# Sequential curriculum training
# =========================================================================

def train_hrl_curriculum_sequential(
    agent: PPOAgentHRL,
    scenario_sequence: list,
    env_factory,
    episodes_per_stage: int = 120,
    num_trajectories_per_update: int = 10,
    delta_actions: bool = True,
    randomize: bool = True,
    val_freq: int = 10,
    num_val_episodes: int = 5,
    save_dir: str = None,
    use_wandb: bool = True,
    reward_rescaling: bool = True,
    rescale_warmup: int = 5,
):
    """
    Sequential curriculum: train on scA for N eps, then scB, …, then scE.

    At the start of each new stage, the agent keeps its weights from the
    previous stage (warm-start).  Validation still runs across ALL scenarios
    to track generalisation.

    Returns:
        return_list, final_return  (same signature as mixed variant)
    """
    if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({"curriculum_type": "sequential"})

    # Cache envs
    envs = {}
    scenario_agent_ids = {}
    for sc in scenario_sequence:
        envs[sc] = env_factory(sc)
        scenario_agent_ids[sc] = envs[sc].possible_agents[0]

    all_returns = []
    global_episode = 0
    global_update = 0
    best_avg_return = float("-inf")

    for stage_idx, sc in enumerate(scenario_sequence):
        env = envs[sc]
        agent_id = scenario_agent_ids[sc]

        print(f"\n{'=' * 60}")
        print(
            f"Sequential Stage {stage_idx + 1}/{len(scenario_sequence)}: {sc}"
        )
        print(f"  agent_id: {agent_id}")
        print(f"  episodes this stage: {episodes_per_stage}")
        print(f"{'=' * 60}")

        # Per-scenario reward scaler (only this scenario active)
        reward_scaler = ScenarioRewardScaler(
            [sc], target_std=1.0, warmup_episodes=rescale_warmup
        )

        agent.init_batch_buffer()
        batch_scenario_tags = []
        stage_returns = []

        if hasattr(agent, "total_updates"):
            effective_updates = max(
                1,
                int(
                    episodes_per_stage
                    / float(max(1, num_trajectories_per_update))
                    * 0.8
                ),
            )
            agent.total_updates = effective_updates

        for ep in range(episodes_per_stage):
            agent.reset_buffer()

            if global_episode == 0:
                obs, infos = env.reset(options={"randomize": False})
            else:
                obs, infos = env.reset(options={"randomize": randomize})

            episode_return = 0.0
            done = False

            while not done:
                agent_state = obs[agent_id]
                action, duration, mu, sigma, dur_probs = agent.take_action(
                    agent_state, return_distribution=True
                )

                if delta_actions:
                    from rl.rl_utils import extract_current_gate_widths
                    absolute_action = (
                        extract_current_gate_widths(obs[agent_id], agent.act_dim)
                        + action
                    )
                    absolute_action = np.clip(
                        absolute_action, agent.act_low, agent.act_high
                    )
                else:
                    absolute_action = action

                cumul_reward = 0.0
                start_obs = obs[agent_id].copy()

                for k_step in range(duration):
                    if done:
                        break
                    next_obs, rewards, terms, truncs, infos = env.step(
                        {agent_id: absolute_action}
                    )
                    gamma = agent.gamma
                    cumul_reward += rewards[agent_id] * (gamma ** k_step)
                    obs = next_obs
                    done = any(terms.values()) or any(truncs.values())

                agent.store_transition(
                    state=start_obs,
                    action=action,
                    next_state=obs[agent_id],
                    reward=cumul_reward,
                    done=done,
                    duration=duration,
                )
                episode_return += cumul_reward

            agent.store_trajectory()
            batch_scenario_tags.append(sc)
            reward_scaler.add_return(sc, episode_return)

            all_returns.append(episode_return)
            stage_returns.append(episode_return)
            global_episode += 1

            # ── Batch update ──────────────────────────────────────────
            if agent.get_batch_size() >= num_trajectories_per_update:
                if reward_rescaling:
                    reward_scaler.rescale_batch_buffer(
                        agent.batch_buffer, batch_scenario_tags
                    )

                if hasattr(env, "ret_rms") and env.ret_rms is not None:
                    try:
                        agent.set_reward_normalizer_var(float(env.ret_rms.var))
                    except Exception:
                        pass
                agent.update_batch()

                global_update += 1
                batch_scenario_tags = []

                # ── WandB ─────────────────────────────────────────────
                if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
                    wandb.log(
                        {
                            "update": global_update,
                            "episode": global_episode,
                            "stage": stage_idx,
                            "stage_scenario": sc,
                            "batch_avg_return": float(
                                np.mean(stage_returns[-num_trajectories_per_update:])
                            ),
                        }
                    )

                # ── Validation on ALL scenarios ───────────────────────
                if save_dir and global_update % val_freq == 0:
                    val_returns = []
                    for val_sc in scenario_sequence:
                        val_env = envs[val_sc]
                        val_aid = scenario_agent_ids[val_sc]
                        val_result = validate_agents(
                            val_env,
                            {val_aid: agent},
                            delta_actions=delta_actions,
                            num_episodes=num_val_episodes,
                            randomize=True,
                        )
                        val_returns.append(val_result["avg_return"])

                        if (
                            use_wandb
                            and WANDB_AVAILABLE
                            and wandb.run is not None
                        ):
                            wandb.log(
                                {
                                    f"val_{val_sc}_return": val_result[
                                        "avg_return"
                                    ],
                                    "val_update": global_update,
                                }
                            )

                    avg_val = np.mean(val_returns)
                    if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
                        wandb.log(
                            {
                                "val_avg_all_scenarios": avg_val,
                                "val_update": global_update,
                            }
                        )

                    if avg_val > best_avg_return:
                        best_avg_return = avg_val
                        canonical_aid = scenario_agent_ids[
                            scenario_sequence[0]
                        ]
                        save_all_agents(
                            {canonical_aid: agent},
                            save_dir,
                            metadata={
                                "episode": int(global_episode),
                                "update": int(global_update),
                                "stage": stage_idx,
                                "val_avg_all_scenarios": float(avg_val),
                                "val_per_scenario": {
                                    s: float(r)
                                    for s, r in zip(
                                        scenario_sequence, val_returns
                                    )
                                },
                                "scenarios": scenario_sequence,
                            },
                        )
                        print(
                            f"\n[Val] New best avg return: {best_avg_return:.3f}"
                            f" at update {global_update} (stage {stage_idx})"
                        )

            # Progress
            if (ep + 1) % 10 == 0:
                avg_r = (
                    np.mean(stage_returns[-10:]) if stage_returns else 0.0
                )
                print(
                    f"  [{sc}] ep {ep + 1}/{episodes_per_stage}"
                    f"  avg_ret={avg_r:.3f}"
                )

    final_return = all_returns[-1] if all_returns else 0.0
    return all_returns, final_return


# =========================================================================
# Single experiment runner
# =========================================================================

def run_single_experiment(cfg: dict, test_run: bool = False):
    """
    Run a single training + evaluation experiment for the given config dict.
    """
    tag = cfg["experiment_tag"]
    cur_type = cfg["curriculum_type"]
    seed = cfg["seed"]

    # Override for smoke-test
    if test_run:
        cfg["num_episodes"] = 10
        cfg["num_traj_per_update"] = 2
        cfg["val_freq"] = 2
        cfg["num_val_episodes"] = 1

    print("\n" + "=" * 70)
    print(f"EXPERIMENT:  {tag}")
    print(f"  curriculum : {cur_type}")
    print(f"  lstm_layers: {cfg['num_lstm_layers']}")
    print(f"  num_heads  : {cfg['num_heads']}")
    print(f"  hidden_size: {cfg['lstm_hidden_size']}")
    print(f"  state_opt  : {cfg['state_option']}")
    print(f"  value_fus  : {cfg['value_fusion']}")
    print(f"  episodes   : {cfg['num_episodes']}")
    print("=" * 70)

    # ── Seed ───────────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    # ── WandB init ─────────────────────────────────────────────────────────
    use_wandb = WANDB_AVAILABLE and not test_run
    if use_wandb:
        wandb.init(
            project="crowd-control-curriculum-exp",
            name=tag,
            config=cfg,
            reinit=True,
        )

    # ── Environment factory ────────────────────────────────────────────────
    state_option = cfg["state_option"]

    def env_factory(dataset: str):
        return create_env(
            dataset,
            obs_mode=state_option,
            normalize_obs=False,
            norm_reward=True,
            action_gap=1,
        )

    # ── Create agent (dims from first scenario) ────────────────────────────
    ref_env = env_factory(ALL_SCENARIOS[0])
    ref_env.seed(seed)
    ref_aid = ref_env.possible_agents[0]

    agent = PPOAgentHRL(
        obs_dim=ref_env.observation_space(ref_aid).shape[0],
        act_dim=ref_env.action_space(ref_aid).shape[0],
        act_low=ref_env.action_space(ref_aid).low,
        act_high=ref_env.action_space(ref_aid).high,
        actor_lr=cfg["actor_lr"],
        critic_lr=cfg["critic_lr"],
        gamma=cfg["gamma"],
        lmbda=cfg["lmbda"],
        entropy_coef=cfg["entropy_coef"],
        kl_tolerance=cfg["kl_tolerance"],
        use_delta_actions=cfg["use_delta_actions"],
        max_delta=cfg["max_delta"],
        lstm_hidden_size=cfg["lstm_hidden_size"],
        num_lstm_layers=cfg["num_lstm_layers"],
        num_heads=cfg["num_heads"],
        use_param_noise=cfg["use_param_noise"],
        use_action_noise=cfg["use_action_noise"],
        num_episodes=cfg["num_episodes"],
        tm_window=cfg["tm_window"],
        max_duration=cfg["max_duration"],
        duration_entropy_coef=cfg["duration_entropy_coef"],
        duration_entropy_coef_min=cfg["duration_entropy_coef_min"],
        value_fusion=cfg["value_fusion"],
        use_lr_decay=cfg["use_lr_decay"],
    )
    del ref_env

    if "new_reward" in tag:
        save_dir = f"./rl/analysis/checkpoints/{tag}"
    else:
        save_dir = f"./checkpoints/experiments/{tag}"

    # ── Train ──────────────────────────────────────────────────────────────
    t0 = time.time()

    if cur_type == "mixed":
        return_list, final_return = train_hrl_curriculum_mixed(
            agent=agent,
            scenario_sequence=ALL_SCENARIOS,
            env_factory=env_factory,
            num_episodes=cfg["num_episodes"],
            num_trajectories_per_update=cfg["num_traj_per_update"],
            delta_actions=cfg["use_delta_actions"],
            randomize=True,
            val_freq=cfg["val_freq"],
            num_val_episodes=cfg["num_val_episodes"],
            save_dir=save_dir,
            use_wandb=use_wandb,
            scenario_sampling=cfg["scenario_sampling"],
            reward_rescaling=cfg["reward_rescaling"],
            rescale_warmup=cfg["rescale_warmup"],
        )
    elif cur_type == "sequential":
        episodes_per_stage = cfg["num_episodes"] // len(ALL_SCENARIOS)
        return_list, final_return = train_hrl_curriculum_sequential(
            agent=agent,
            scenario_sequence=ALL_SCENARIOS,
            env_factory=env_factory,
            episodes_per_stage=episodes_per_stage,
            num_trajectories_per_update=cfg["num_traj_per_update"],
            delta_actions=cfg["use_delta_actions"],
            randomize=True,
            val_freq=cfg["val_freq"],
            num_val_episodes=cfg["num_val_episodes"],
            save_dir=save_dir,
            use_wandb=use_wandb,
            reward_rescaling=cfg["reward_rescaling"],
            rescale_warmup=cfg["rescale_warmup"],
        )
    elif cur_type == "incremental":
        return_list, final_return = train_hrl_curriculum_incremental(
            agent=agent,
            scenario_sequence=ALL_SCENARIOS,
            env_factory=env_factory,
            total_episodes=cfg["num_episodes"],
            num_trajectories_per_update=cfg["num_traj_per_update"],
            delta_actions=cfg["use_delta_actions"],
            randomize=True,
            val_freq=cfg["val_freq"],
            num_val_episodes=cfg["num_val_episodes"],
            save_dir=save_dir,
            use_wandb=use_wandb,
            reward_rescaling=cfg["reward_rescaling"],
            rescale_warmup=cfg["rescale_warmup"],
            threshold_return=cfg["threshold_return"],
            min_episodes_per_stage=cfg["min_episodes_per_stage"],
        )
    else:
        raise ValueError(f"Unknown curriculum type: {cur_type}")

    train_time = time.time() - t0

    # ── Final evaluation (load best checkpoint) ────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"Final evaluation for {tag}  (train time: {train_time / 60:.1f} min)")
    print(f"{'─' * 60}")

    eval_results = {}
    try:
        loaded_agents, _ = load_all_agents(save_dir=save_dir, device="cpu")
        loaded_agent = list(loaded_agents.values())[0]

        for eval_sc in EVAL_SCENARIOS:
            eval_env = env_factory(eval_sc)
            eval_aid = eval_env.possible_agents[0]
            result = evaluate_agents(
                eval_env,
                {eval_aid: loaded_agent},
                delta_actions=cfg["use_delta_actions"],
                deterministic=True,
                seed=42,
                randomize=True,
                num_runs=5,
                verbose=False,
            )
            eval_results[eval_sc] = {
                "avg_reward": float(result["avg_reward"]),
                "avg_reward_std": float(result["avg_reward_std"]),
            }
            print(
                f"  {eval_sc}: {result['avg_reward']:.3f}"
                f" ± {result['avg_reward_std']:.3f}"
            )

            if use_wandb and wandb.run is not None:
                short = eval_sc.split("_")[-1]
                wandb.log(
                    {
                        f"eval_{short}_reward": result["avg_reward"],
                        f"eval_{short}_reward_std": result["avg_reward_std"],
                    }
                )
    except Exception as e:
        print(f"  [!] Could not load best model for evaluation: {e}")

    # ── Save experiment summary ────────────────────────────────────────────
    summary = {
        "experiment_tag": tag,
        "config": cfg,
        "train_time_seconds": train_time,
        "final_return": float(np.asarray(final_return).item()),
        "eval_results": eval_results,
        "timestamp": datetime.now().isoformat(),
    }
    summary_path = os.path.join(save_dir, "experiment_summary.json")
    os.makedirs(save_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary saved to {summary_path}")

    if use_wandb and wandb.run is not None:
        wandb.finish()

    return summary


# =========================================================================
# CLI entrypoint
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run curriculum training experiments for PPO-HRL"
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Smoke-test with very few episodes, no WandB",
    )
    parser.add_argument(
        "--config-index",
        type=int,
        default=None,
        help="Run only the config at this 0-based index",
    )
    parser.add_argument(
        "--curriculum",
        type=str,
        default=None,
        choices=["mixed", "sequential", "incremental"],
        help="Filter grid to a single curriculum type",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="Print the experiment grid and exit",
    )
    args = parser.parse_args()

    configs = build_experiment_grid(curriculum_filter=args.curriculum)

    if args.list_configs:
        print(f"Total experiments: {len(configs)}\n")
        for i, c in enumerate(configs):
            print(f"  [{i:2d}] {c['experiment_tag']}")
        return

    # Select configs to run
    if args.config_index is not None:
        if args.config_index < 0 or args.config_index >= len(configs):
            print(f"Error: config-index must be 0..{len(configs) - 1}")
            return
        configs_to_run = [configs[args.config_index]]
    else:
        configs_to_run = configs

    print(f"Running {len(configs_to_run)} experiment(s)")
    if args.test_run:
        print("  ⚡ TEST-RUN mode (tiny episode count, no WandB)")

    all_summaries = []
    for i, cfg in enumerate(configs_to_run):
        print(f"\n{'▓' * 70}")
        print(f"  Experiment {i + 1}/{len(configs_to_run)}")
        print(f"{'▓' * 70}")
        summary = run_single_experiment(cfg, test_run=args.test_run)
        all_summaries.append(summary)

    # ── Print leaderboard ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT LEADERBOARD")
    print("=" * 70)
    print(f"{'Tag':<55} {'Avg Eval':>10}")
    print("-" * 70)
    for s in sorted(
        all_summaries,
        key=lambda x: np.mean(
            [v["avg_reward"] for v in x["eval_results"].values()]
        )
        if x["eval_results"]
        else float("-inf"),
        reverse=True,
    ):
        if s["eval_results"]:
            avg_eval = np.mean(
                [v["avg_reward"] for v in s["eval_results"].values()]
            )
            print(f"  {s['experiment_tag']:<55} {avg_eval:>10.3f}")
        else:
            print(f"  {s['experiment_tag']:<55} {'N/A':>10}")
    print("=" * 70)


if __name__ == "__main__":
    main()
