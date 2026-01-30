# -*- coding: utf-8 -*-
"""
Evaluate and visualize trained RLlib policies.

Loads a trained RLlib checkpoint and runs episodes to evaluate performance
and optionally render/save visualizations.
"""

import sys
from pathlib import Path
import os

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import argparse
import numpy as np
import ray
import torch
from pathlib import Path
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.core.rl_module import RLModule
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


def evaluate_policy(checkpoint_path, dataset="nine_intersections", num_episodes=5, 
                    render=False, save_simulation=False):
    """
    Evaluate a trained RLlib policy.
    
    Args:
        checkpoint_path: Path to RLlib checkpoint directory
        dataset: Network dataset name
        num_episodes: Number of episodes to evaluate
        render: Whether to render episodes
        save_simulation: Whether to save simulation data
    """
    print("=" * 70)
    print("Evaluating RLlib Policy")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset}")
    print(f"Episodes: {num_episodes}")
    print("-" * 70)
    
    # Initialize Ray with runtime environment to ensure workers can import rl module
    if not ray.is_initialized():
        # Set up runtime environment to include project root in Python path
        runtime_env = {
            "env_vars": {
                "PYTHONPATH": str(project_root)
            },
            "working_dir": str(project_root)
        }
        ray.init(
            ignore_reinit_error=True, 
            log_to_driver=False,
            runtime_env=runtime_env
        )
    
    # Convert checkpoint path to absolute Path object
    checkpoint_path_obj = Path(checkpoint_path)
    if not checkpoint_path_obj.is_absolute():
        # Make relative to project root
        checkpoint_path_obj = Path(project_root) / checkpoint_path
    
    # Find the directory containing rllib_checkpoint.json
    checkpoint_dir = None
    
    # Check if the path itself contains rllib_checkpoint.json
    if checkpoint_path_obj.is_dir() and (checkpoint_path_obj / "rllib_checkpoint.json").exists():
        checkpoint_dir = checkpoint_path_obj
    elif checkpoint_path_obj.is_dir():
        # Look for subdirectories with checkpoints (e.g., final_model, best_model, checkpoint_XXXX)
        for subdir in checkpoint_path_obj.iterdir():
            if subdir.is_dir() and (subdir / "rllib_checkpoint.json").exists():
                checkpoint_dir = subdir
                break
        
        if checkpoint_dir is None:
            # List available checkpoints for user
            print(f"✗ Error: No checkpoint found at {checkpoint_path_obj}")
            print(f"  Looking for directory containing 'rllib_checkpoint.json'")
            print(f"  Available subdirectories:")
            for subdir in checkpoint_path_obj.iterdir():
                if subdir.is_dir():
                    has_checkpoint = (subdir / "rllib_checkpoint.json").exists()
                    print(f"    - {subdir.name} {'✓ (has checkpoint)' if has_checkpoint else ''}")
            raise ValueError(f"No valid checkpoint found in {checkpoint_path_obj}")
    elif checkpoint_path_obj.is_file():
        # If it's a file, use parent directory
        checkpoint_dir = checkpoint_path_obj.parent
        if not (checkpoint_dir / "rllib_checkpoint.json").exists():
            raise ValueError(f"Parent directory does not contain rllib_checkpoint.json: {checkpoint_dir}")
    else:
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path_obj}")
    
    # Convert to absolute string path (RLlib expects a string, not Path object)
    checkpoint_dir_str = str(checkpoint_dir.resolve())
    
    # Load trained algorithm
    print(f"Loading checkpoint from: {checkpoint_dir_str}")
    algo = PPO.from_checkpoint(checkpoint_dir_str)
    print("✓ Checkpoint loaded")

    # Policy mapping function - should match the one used during training
    # For independent policies (default in training), each agent_id maps to itself
    def policy_mapping_fn(agent_id, *args, **kwargs):
        # Independent policies: each agent has its own policy
        return agent_id

        # Alternative configurations (comment/uncomment as needed):
        # Shared policy: all agents use same policy
        # return "shared_policy"

        # Type-based policies: separators vs gaters
        # return "sep_policy" if agent_id.startswith("sep_") else "gate_policy"

    # Load RLModules for each agent's policy (official RLlib method)
    # See: https://docs.ray.io/en/latest/rllib/getting-started.html
    print("Loading RL modules for each policy...")
    rl_modules = {}
    checkpoint_path = Path(checkpoint_dir_str)
    # Create base environment directly for evaluation
    base_env = PedNetParallelEnv(
        dataset=dataset,
        normalize_obs=True,
        with_density_obs=True
    )
    
    # Also create wrapped version for compatibility
    env = ParallelPettingZooEnv(base_env)

    for agent_id in base_env.possible_agents:
        policy_id = policy_mapping_fn(agent_id)
        
        # Skip if we already loaded this policy
        if policy_id in rl_modules:
            continue

        try:
            # Official RLlib method: Load RLModule from checkpoint path
            # Path structure: checkpoint/learner_group/learner/rl_module/[policy_id]
            module_path = (
                checkpoint_path
                / "learner_group"
                / "learner"
                / "rl_module"
                / policy_id
            )
            
            if module_path.exists():
                rl_modules[policy_id] = RLModule.from_checkpoint(module_path)
                print(f"  ✓ Loaded module for policy: {policy_id}")
            else:
                print(f"  ✗ Module path not found: {module_path}")
                raise FileNotFoundError(f"Module not found at {module_path}")
        except Exception as e:
            print(f"  ✗ Failed to load module for {policy_id}: {e}")
            print(f"    Trying alternative method...")
            
            # Fallback: try to get module from algorithm
            try:
                rl_modules[policy_id] = algo.get_module(policy_id)
                print(f"  ✓ Loaded module via algo.get_module()")
            except Exception as e2:
                print(f"  ✗ Could not load module: {e2}")
                raise ValueError(f"Failed to load RL module for policy {policy_id}")

    print(f"✓ Loaded {len(rl_modules)} RL modules")
    
    
    print(f"✓ Environment created")
    print(f"  - Agents: {len(base_env.possible_agents)}")
    
    # Run evaluation episodes
    print("\n" + "=" * 70)
    print("Running Evaluation Episodes")
    print("=" * 70)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, infos = base_env.reset()
        episode_reward = {agent_id: 0.0 for agent_id in base_env.possible_agents}
        episode_length = 0
        done = False
        
        while not done:
            # Get actions from trained policies using RL modules (official RLlib method)
            # Reference: https://docs.ray.io/en/latest/rllib/getting-started.html
            actions = {}
            for agent_id in base_env.possible_agents:
                # Map agent_id to policy_id
                policy_id = policy_mapping_fn(agent_id)
                rl_module = rl_modules[policy_id]

                # Compute the next action from a batch (B=1) of observations
                obs_batch = torch.from_numpy(obs[agent_id]).unsqueeze(0)  # add batch dimension
                model_outputs = rl_module.forward_inference({"obs": obs_batch})

                # Extract the action distribution parameters and dissolve batch dimension
                action_dist_params = model_outputs["action_dist_inputs"][0].numpy()

                # For continuous actions (Box spaces), take the mean (max likelihood)
                # PPO uses Gaussian distributions: [mean, log_stddev]
                # For deterministic evaluation, use the mean (index 0:action_dim)
                action_dim = base_env.action_space(agent_id).shape[0]
                greedy_action = action_dist_params[0:action_dim]  # Take the mean part

                # Clip to action space bounds
                action = np.clip(
                    greedy_action,
                    a_min=base_env.action_space(agent_id).low,
                    a_max=base_env.action_space(agent_id).high,
                )

                actions[agent_id] = action
            
            # Step environment
            obs, rewards, terms, truncs, infos = base_env.step(actions)
            
            # Accumulate rewards
            for agent_id, reward in rewards.items():
                episode_reward[agent_id] += reward
            
            episode_length += 1
            
            # Check if episode is done
            done = any(terms.values()) or any(truncs.values())
        
        # Store episode statistics
        avg_reward = np.mean(list(episode_reward.values()))
        episode_rewards.append(avg_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}/{num_episodes}: "
              f"reward={avg_reward:8.3f}, length={episode_length:4d}")
        
        # Optionally save simulation
        if save_simulation:
            save_dir = f"outputs/rllib_eval_{dataset}_ep{episode}"
            base_env.save(save_dir)
            print(f"  → Simulation saved to {save_dir}")
        
        # Optionally render (only for last episode to save time)
        if render and episode == num_episodes - 1:
            print("  → Rendering animation...")
            base_env.render_mode = "animate"
            sim_dir = f"outputs/rllib_eval_{dataset}_ep{episode}"
            base_env.render(
                simulation_dir=sim_dir,
                variable='density',
                vis_actions=True
            )
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    print(f"Episodes: {num_episodes}")
    print(f"Average reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Min reward: {np.min(episode_rewards):.3f}")
    print(f"Max reward: {np.max(episode_rewards):.3f}")
    print("=" * 70)
    
    # Cleanup
    algo.stop()
    ray.shutdown()
    
    return episode_rewards, episode_lengths


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained RLlib policy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate final_model checkpoint
  python eval_rllib.py --checkpoint rl/rl_models_rllib/nine_intersections/final_model
  
  # Evaluate with rendering
  python eval_rllib.py --checkpoint rl/rl_models_rllib/nine_intersections/final_model --render
  
  # Evaluate and save simulation data
  python eval_rllib.py --checkpoint rl/rl_models_rllib/nine_intersections/final_model --save
        """
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to RLlib checkpoint directory (e.g., rl/rl_models_rllib/nine_intersections/final_model)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="nine_intersections",
        help="Network dataset"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the last episode"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save simulation data for each episode"
    )
    
    args = parser.parse_args()
    
    # Evaluate
    evaluate_policy(
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        num_episodes=args.episodes,
        render=args.render,
        save_simulation=args.save
    )


if __name__ == "__main__":
    main()

