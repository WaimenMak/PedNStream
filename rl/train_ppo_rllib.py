# -*- coding: utf-8 -*-
"""
Train PPO agents using Ray RLlib on PedNetParallelEnv.

Simple training script for multi-agent reinforcement learning on pedestrian
traffic control using RLlib's distributed training capabilities.
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import argparse
from datetime import datetime
import json
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from rl import PedNetParallelEnv


class PedNetRLlibEnv(ParallelPettingZooEnv):
    """RLlib-compatible wrapper for PedNetParallelEnv."""
    
    def __init__(self, config=None):
        config = config or {}
        base_env = PedNetParallelEnv(
            dataset=config.get("dataset", "nine_intersections"),
            normalize_obs=config.get("normalize_obs", True),
            with_density_obs=config.get("with_density_obs", True)
        )
        super().__init__(base_env)


def train_rllib_ppo(
    dataset="nine_intersections",
    num_iterations=50,
    checkpoint_freq=10,
    num_workers=2,
    train_batch_size=4000,
    minibatch_size=128,
    save_dir=None
):
    """
    Train PPO agents using RLlib.
    
    Args:
        dataset: Network dataset name
        num_iterations: Number of training iterations
        checkpoint_freq: Save checkpoint every N iterations
        num_workers: Number of parallel rollout workers
        train_batch_size: Training batch size
        minibatch_size: Minibatch size
        save_dir: Directory to save checkpoints and logs
    """
    print("=" * 70)
    print("Training PPO with Ray RLlib")
    print("=" * 70)
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=num_workers + 1, log_to_driver=False)
        print(f"✓ Ray initialized with {num_workers} workers")
    
    # Setup save directory
    if save_dir is None:
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # save_dir = Path(f"rl_models_rllib/{dataset}_{timestamp}")
        save_dir = Path(f"rl_models_rllib/{dataset}")
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Save directory: {save_dir}")
    
    # Get agent IDs from environment
    test_env = PedNetRLlibEnv({"dataset": dataset})
    agent_ids = test_env.get_agent_ids()
    print(f"\n✓ Environment: {dataset}")
    print(f"  - Agents: {len(agent_ids)}")
    #enumerate the agent ids
    print("  - Agent IDs:")
    for i, agent_id in enumerate(agent_ids):
        print(f"    {i}: {agent_id}")
    
    # Configure PPO
    print("\n" + "=" * 70)
    print("Configuring PPO Algorithm")
    print("=" * 70)
    
    config = (
        PPOConfig()
        .environment(
            env=PedNetRLlibEnv,
            env_config={
                "dataset": dataset,
                "normalize_obs": True,
                "with_density_obs": True
            }
        )
        .framework("torch")
        .multi_agent(
            # Each agent gets its own independent policy
            policies={agent_id: (None, None, None, {}) for agent_id in agent_ids},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
        )
        .env_runners(
            num_env_runners=num_workers,
            rollout_fragment_length=200,
        )
        .training(
            train_batch_size=train_batch_size,
            minibatch_size=minibatch_size,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
        )
        .evaluation(
            evaluation_interval=None,  # Disable evaluation for now
        )
        .reporting(
            min_sample_timesteps_per_iteration=1000,
        )
    )
    
    print(f"✓ Configuration:")
    print(f"  - Workers: {num_workers}")
    print(f"  - Train batch size: {train_batch_size}")
    print(f"  - Policies: {len(agent_ids)} (independent)")
    print(f"  - Framework: torch")
    
    # Build algorithm
    print("\n" + "=" * 70)
    print("Building Algorithm")
    print("=" * 70)
    algo = config.build()
    print("✓ PPO algorithm built successfully")
    
    # Save configuration
    config_file = save_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump({
            "dataset": dataset,
            "num_iterations": num_iterations,
            "num_workers": num_workers,
            "train_batch_size": train_batch_size,
            "agent_ids": list(agent_ids),
            "num_agents": len(agent_ids)
        }, f, indent=2)
    print(f"✓ Configuration saved to {config_file}")
    
    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    print(f"Iterations: {num_iterations}")
    print(f"Checkpoint frequency: every {checkpoint_freq} iterations")
    print("-" * 70)
    
    results_log = []
    best_reward = float('-inf')
    
    try:
        for iter_num in range(1, num_iterations + 1):
            # Train one iteration
            result = algo.train()
            
            # Extract metrics
            module_returns = result.get('env_runners', {}).get('module_episode_returns_mean', {})
            if module_returns:
                episode_reward_mean = sum(module_returns.values()) / len(module_returns)
            else:
                episode_reward_mean = 0.0
            episodes_this_iter = int(result.get('env_runners', {}).get('num_episodes', 0) or 0)
            timesteps_total = int(result.get('num_env_steps_sampled_lifetime', 0) or 0)
            
            # Log progress
            print(f"Iter {iter_num}/{num_iterations} | "
                  f"Reward: {episode_reward_mean:8.3f} | "
                  f"Episodes: {episodes_this_iter:3d} | "
                  f"Timesteps: {timesteps_total:6d}")
            
            # Store results
            results_log.append({
                "iteration": iter_num,
                "episode_reward_mean": episode_reward_mean,
                "episodes_this_iter": episodes_this_iter,
                "timesteps_total": timesteps_total
            })
            
            # Save checkpoint
            if iter_num % checkpoint_freq == 0 or iter_num == num_iterations:
                checkpoint_path = save_dir / f"checkpoint_{iter_num:04d}"
                algo.save(str(checkpoint_path.resolve()) if hasattr(checkpoint_path, "resolve") else str(checkpoint_path))
                print(f"  → Checkpoint saved: {checkpoint_path}")
                
                # Save best model
                if episode_reward_mean > best_reward:
                    best_reward = episode_reward_mean
                    best_path = save_dir / "best_model"
                    algo.save(str(best_path.resolve()) if hasattr(best_path, "resolve") else str(best_path))
                    print(f"  → Best model updated: {best_path}")
            
            # Save results log
            results_file = save_dir / "training_results.json"
            with open(results_file, 'w') as f:
                json.dump(results_log, f, indent=2)
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Final save
        final_path = save_dir / "final_model"
        # Fix for pyarrow.lib.ArrowInvalid: URI has empty scheme:
        # Ensure the final_path is a string (and resolve any weird path-like object issues)
        path_str = str(final_path.resolve()) if hasattr(final_path, "resolve") else str(final_path)
        algo.save(path_str)
        print(f"\n✓ Final model saved: {final_path}")
        
        # Cleanup
        algo.stop()
        ray.shutdown()
        print("✓ Ray shutdown")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)
    print(f"Total iterations: {len(results_log)}")
    if results_log:
        rewards = [r["episode_reward_mean"] for r in results_log]
        print(f"Best reward: {max(rewards):.3f}")
        print(f"Final reward: {rewards[-1]:.3f}")
        print(f"Average reward (last 10): {sum(rewards[-10:]) / min(10, len(rewards)):.3f}")
    print(f"\nResults saved to: {save_dir}")
    print("=" * 70)
    
    return algo, results_log


def main():
    parser = argparse.ArgumentParser(description="Train PPO on PedNet using RLlib")
    parser.add_argument(
        "--dataset",
        type=str,
        default="45_intersections",
        help="Network dataset (nine_intersections, long_corridor, etc.)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of training iterations"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel rollout workers"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size"
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=2000,
        help="Training buffer size"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10000,
        help="Save checkpoint every N iterations"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save models (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    # Train
    train_rllib_ppo(
        dataset=args.dataset,
        num_iterations=args.iterations,
        checkpoint_freq=args.checkpoint_freq,
        num_workers=args.workers,
        train_batch_size=args.buffer_size,
        minibatch_size=args.batch_size,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()

