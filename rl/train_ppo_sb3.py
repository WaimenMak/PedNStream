# -*- coding: utf-8 -*-
"""
Train PPO agents using Stable-Baselines3 library.

This script uses the industry-standard SB3 library with SuperSuit wrappers
to train agents on the PedNetParallelEnv environment.
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import numpy as np
from rl import PedNetParallelEnv
from datetime import datetime
import os
from gymnasium import spaces as gym_spaces

# Import SB3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
except ImportError:
    print("ERROR: Missing dependencies. Please install:")
    print("  pip install stable-baselines3")
    sys.exit(1)


class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to tensorboard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        # Log episode statistics
        if len(self.episode_rewards) > 0:
            self.logger.record("rollout/ep_rew_mean", np.mean(self.episode_rewards[-100:]))
            self.logger.record("rollout/ep_len_mean", np.mean(self.episode_lengths[-100:]))
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout."""
        pass


class PedNetSB3Wrapper:
    """
    Wrapper to convert PedNetParallelEnv to single-agent format for SB3.
    
    Treats all agents as one by concatenating observations and actions.
    """
    def __init__(self, env):
        self.env = env
        self.agents = env.possible_agents
        
        # Build concatenated observation space
        obs_spaces = [env.observation_space(agent) for agent in self.agents]
        total_obs_dim = sum(space.shape[0] for space in obs_spaces)
        self.observation_space = gym_spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_dim,),
            dtype=np.float32
        )
        
        # Build concatenated action space
        act_spaces = [env.action_space(agent) for agent in self.agents]
        total_act_dim = sum(space.shape[0] for space in act_spaces)
        act_lows = np.concatenate([space.low for space in act_spaces])
        act_highs = np.concatenate([space.high for space in act_spaces])
        self.action_space = gym_spaces.Box(
            low=act_lows,
            high=act_highs,
            shape=(total_act_dim,),
            dtype=np.float32
        )
        
        # Store dimensions for splitting actions
        self.act_dims = [space.shape[0] for space in act_spaces]
        self.obs_dims = [space.shape[0] for space in obs_spaces]
    
    def reset(self, seed=None, options=None):
        """Reset environment and return concatenated observation."""
        obs_dict, infos = self.env.reset(seed=seed, options=options)
        
        # Concatenate observations
        obs_list = [obs_dict[agent] for agent in self.agents]
        concat_obs = np.concatenate(obs_list, dtype=np.float32)
        
        # Combine infos (just use first agent's info for simplicity)
        combined_info = infos[self.agents[0]] if self.agents else {}
        
        return concat_obs, combined_info
    
    def step(self, action):
        """Step environment with split actions."""
        # Split concatenated action into individual agent actions
        actions_dict = {}
        start_idx = 0
        for i, agent in enumerate(self.agents):
            end_idx = start_idx + self.act_dims[i]
            actions_dict[agent] = action[start_idx:end_idx]
            start_idx = end_idx
        
        # Step environment
        obs_dict, rewards, terms, truncs, infos = self.env.step(actions_dict)
        
        # Concatenate observations
        obs_list = [obs_dict[agent] for agent in self.agents]
        concat_obs = np.concatenate(obs_list, dtype=np.float32)
        
        # Sum rewards (cooperative task)
        total_reward = sum(rewards.values())
        
        # Any agent done = episode done
        done = any(terms.values()) or any(truncs.values())
        terminated = any(terms.values())
        truncated = any(truncs.values())
        
        # Combine infos
        combined_info = infos[self.agents[0]] if self.agents else {}
        combined_info['individual_rewards'] = rewards
        
        return concat_obs, total_reward, terminated, truncated, combined_info
    
    def render(self):
        """Render environment."""
        return self.env.render()
    
    def close(self):
        """Close environment."""
        return self.env.close()


def make_env(dataset="long_corridor", normalize_obs=True, with_density_obs=True, 
             randomize=False, seed=None):
    """
    Create and wrap a PedNet environment for SB3.
    
    Args:
        dataset: Network dataset name
        normalize_obs: Whether to normalize observations
        with_density_obs: Whether to include density in observations
        randomize: Whether to randomize network at each reset
        seed: Random seed
    
    Returns:
        Wrapped environment compatible with SB3
    """
    def _init():
        env = PedNetParallelEnv(
            dataset=dataset,
            normalize_obs=normalize_obs,
            with_density_obs=with_density_obs
        )
        
        # Wrap with custom wrapper for SB3
        env = PedNetSB3Wrapper(env)
        
        return env
    
    return _init


def train_sb3_ppo(
    dataset=None,
    total_timesteps=100_000,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    normalize_obs=True,
    with_density_obs=True,
    save_dir="rl_models_sb3",
    tensorboard_log="./tensorboard_logs",
    eval_freq=10_000,
    save_freq=20_000,
):
    """
    Train PPO agents using Stable-Baselines3.
    
    Args:
        dataset: Network dataset to use
        total_timesteps: Total training timesteps
        learning_rate: Learning rate
        n_steps: Number of steps per rollout
        batch_size: Minibatch size
        n_epochs: Number of epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_range: PPO clipping parameter
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Max gradient norm for clipping
        normalize_obs: Whether to normalize observations
        with_density_obs: Whether to include density observations
        save_dir: Directory to save models
        tensorboard_log: Directory for tensorboard logs
        eval_freq: Evaluation frequency (timesteps)
        save_freq: Model checkpoint frequency (timesteps)
    """
    print("=" * 70)
    print("Training PPO Agents using Stable-Baselines3")
    print("=" * 70)
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(tensorboard_log, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{dataset}_{timestamp}"
    
    print(f"\nRun name: {run_name}")
    print(f"Dataset: {dataset}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Learning rate: {learning_rate}")
    print(f"Steps per rollout: {n_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Normalize observations: {normalize_obs}")
    print(f"Include density observations: {with_density_obs}")
    
    # Create environment
    print("\nCreating environment...")
    env_fn = make_env(
        dataset=dataset,
        normalize_obs=normalize_obs,
        with_density_obs=with_density_obs,
        randomize=True
    )
    
    # Wrap in DummyVecEnv (SB3 requirement)
    env = DummyVecEnv([env_fn])
    
    # Wrap with monitor for logging
    env = VecMonitor(env)
    
    print(f"Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Create evaluation environment
    print("\nCreating evaluation environment...")
    eval_env_fn = make_env(
        dataset=dataset,
        normalize_obs=normalize_obs,
        with_density_obs=with_density_obs,
        randomize=False  # Deterministic for evaluation
    )
    eval_env = DummyVecEnv([eval_env_fn])
    eval_env = VecMonitor(eval_env)
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=os.path.join(save_dir, run_name),
        name_prefix="ppo_model",
        verbose=1
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, run_name, "best_model"),
        log_path=os.path.join(save_dir, run_name, "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    tensorboard_callback = TensorboardCallback()
    
    callbacks = [checkpoint_callback, eval_callback, tensorboard_callback]
    
    # Create PPO model
    print("\nCreating PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=1,
        tensorboard_log=os.path.join(tensorboard_log, run_name),
        device="auto"  # Automatically use GPU if available
    )
    
    print(f"Model created with policy architecture:")
    print(f"  Policy: MlpPolicy (default: [64, 64] hidden layers)")
    print(f"  Device: {model.device}")
    
    # Train model
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    print("\nTo monitor training progress, run:")
    print(f"  tensorboard --logdir {tensorboard_log}")
    print(f"  Then open: http://localhost:6006\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    # Save final model
    final_model_path = os.path.join(save_dir, run_name, "final_model")
    model.save(final_model_path)
    print(f"\n✅ Final model saved to: {final_model_path}")
    
    # Close environments
    env.close()
    eval_env.close()
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nModel saved to: {save_dir}/{run_name}")
    print(f"Tensorboard logs: {tensorboard_log}/{run_name}")
    print("\nTo load and evaluate the model:")
    print(f'  model = PPO.load("{final_model_path}")')
    print('  obs = env.reset()')
    print('  action, _states = model.predict(obs, deterministic=True)')
    
    return model, run_name


def evaluate_model(model_path, dataset="long_corridor", n_episodes=10, render=False):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to saved model
        dataset: Dataset to evaluate on
        n_episodes: Number of episodes to run
        render: Whether to render the environment
    """
    print("=" * 70)
    print("Evaluating Model")
    print("=" * 70)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = PPO.load(model_path)
    
    # Create environment
    env_fn = make_env(dataset=dataset, normalize_obs=True, with_density_obs=True, randomize=False)
    env = DummyVecEnv([env_fn])
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, dones, info = env.step(action)
            episode_reward += reward[0]  # Extract from vectorized env
            episode_length += 1
            done = dones[0]
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {episode + 1}/{n_episodes}: Reward={episode_reward:.2f}, Length={episode_length}")
    
    env.close()
    
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    
    return episode_rewards, episode_lengths


if __name__ == "__main__":
    # Train PPO agent
    model, run_name = train_sb3_ppo(
        dataset="nine_intersections",
        total_timesteps=200_000,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        normalize_obs=True,
        with_density_obs=True,
        eval_freq=10_000,
        save_freq=20_000
    )
    
    # Optionally evaluate
    # evaluate_model(f"rl_models_sb3/{run_name}/best_model/best_model.zip", n_episodes=5)

