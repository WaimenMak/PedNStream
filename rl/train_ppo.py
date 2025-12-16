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
from rl import PedNetParallelEnv
import torch
from datetime import datetime
import json


class PPOAgent:
    """Simple PPO implementation for continuous action spaces."""
    
    def __init__(self, obs_dim, act_dim, act_low, act_high, lr=3e-4, gamma=0.99, 
                 clip_eps=0.2, entropy_coef=0.01):
        """
        Initialize PPO agent.
        
        Args:
            obs_dim: Observation space dimension
            act_dim: Action space dimension
            act_low: Lower bound of action space
            act_high: Upper bound of action space
            lr: Learning rate
            gamma: Discount factor
            clip_eps: PPO clipping parameter
            entropy_coef: Entropy bonus coefficient
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_low = torch.tensor(act_low, dtype=torch.float32)
        self.act_high = torch.tensor(act_high, dtype=torch.float32)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        
        # Simple MLP policy network
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, act_dim * 2)  # mean and log_std
        )
        
        # Value network
        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value_net.parameters()),
            lr=lr
        )
        
        # Rollout buffer
        self.reset_buffer()
    
    def reset_buffer(self):
        """Clear rollout buffer."""
        self.obs_buffer = []
        self.act_buffer = []
        self.rew_buffer = []
        self.val_buffer = []
        self.logp_buffer = []
    
    def get_action(self, obs, deterministic=False):
        """Sample action from policy."""
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        
        with torch.no_grad():
            policy_out = self.policy(obs_tensor)
            mean = policy_out[:self.act_dim]
            log_std = policy_out[self.act_dim:]
            log_std = torch.clamp(log_std, -20, 2)  # Stability
            
            if deterministic:
                action = mean
            else:
                std = torch.exp(log_std)
                if std.any() < 0:
                    raise ValueError(f"Negative standard deviation: {std.item()}")
                action = torch.normal(mean, std)
            
            # Clip to action bounds
            action = torch.clamp(action, self.act_low, self.act_high)
            
            # Compute log probability
            logp = -0.5 * (((action - mean) / (std + 1e-8)) ** 2 + 
                          2 * log_std + np.log(2 * np.pi))
            logp = logp.sum()
            
            # Compute value
            value = self.value_net(obs_tensor).squeeze()
        
        return action.numpy(), logp.item(), value.item()
    
    def store_transition(self, obs, act, rew, val, logp):
        """Store transition in buffer."""
        self.obs_buffer.append(obs)
        self.act_buffer.append(act)
        self.rew_buffer.append(rew)
        self.val_buffer.append(val)
        self.logp_buffer.append(logp)
    
    def update(self, last_val=0.0, update_epochs=10, batch_size=64):
        """Update policy using PPO."""
        if len(self.obs_buffer) == 0:
            return {}
        
        # Compute returns and advantages
        returns = []
        advantages = []
        ret = last_val
        
        for rew, val in zip(reversed(self.rew_buffer), reversed(self.val_buffer)):
            ret = rew + self.gamma * ret
            adv = ret - val
            returns.insert(0, ret)
            advantages.insert(0, adv)
        
        # Convert to tensors
        obs = torch.tensor(np.array(self.obs_buffer), dtype=torch.float32)
        acts = torch.tensor(np.array(self.act_buffer), dtype=torch.float32)
        old_logps = torch.tensor(self.logp_buffer, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_loss = 0.0
        for _ in range(update_epochs):
            # Forward pass
            policy_out = self.policy(obs)
            mean = policy_out[:, :self.act_dim]
            log_std = policy_out[:, self.act_dim:]
            log_std = torch.clamp(log_std, -20, 2)
            std = torch.exp(log_std)
            
            # Compute new log probabilities
            logps = -0.5 * (((acts - mean) / (std + 1e-8)) ** 2 + 
                           2 * log_std + np.log(2 * np.pi))
            logps = logps.sum(dim=1)
            
            # PPO ratio and clipped objective
            ratio = torch.exp(logps - old_logps)
            clip_adv = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(ratio * advantages, clip_adv).mean()
            
            # Value loss
            values = self.value_net(obs).squeeze()
            value_loss = ((values - returns) ** 2).mean()
            
            # Entropy bonus
            entropy = (log_std + 0.5 * np.log(2 * np.pi * np.e)).sum(dim=1).mean()
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) + list(self.value_net.parameters()),
                max_norm=0.5
            )
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Clear buffer
        self.reset_buffer()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss / update_epochs
        }


def train_ppo(dataset="long_corridor", total_episodes=100, steps_per_episode=None,
              update_freq=200, save_dir="rl_models"):
    """
    Train PPO agents on PedNet environment.
    
    Args:
        dataset: Network dataset to use
        total_episodes: Number of training episodes
        steps_per_episode: Steps per episode (None = use network default)
        update_freq: Update policy every N steps
        save_dir: Directory to save models and logs
    """
    print("=" * 60)
    print("Training PPO Agents on PedNet Environment")
    print("=" * 60)
    
    # Create environment
    env = PedNetParallelEnv(
        dataset=dataset,
        normalize_obs=True,
        with_density_obs=True
    )
    
    steps_per_episode = steps_per_episode or env.simulation_steps
    
    print(f"\nDataset: {dataset}")
    print(f"Episodes: {total_episodes}")
    print(f"Steps per episode: {steps_per_episode}")
    print(f"Update frequency: {update_freq}")
    print(f"\nAgents: {len(env.possible_agents)}")
    for agent_id in env.possible_agents:
        obs_space = env.observation_space(agent_id)
        act_space = env.action_space(agent_id)
        print(f"  - {agent_id}: obs_dim={obs_space.shape[0]}, act_dim={act_space.shape[0]}")
    
    # Create agents
    agents = {}
    for agent_id in env.possible_agents:
        obs_space = env.observation_space(agent_id)
        act_space = env.action_space(agent_id)
        agents[agent_id] = PPOAgent(
            obs_dim=obs_space.shape[0],
            act_dim=act_space.shape[0],
            act_low=act_space.low,
            act_high=act_space.high
        )
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    episode_rewards = {agent_id: [] for agent_id in env.possible_agents}
    global_step = 0
    
    for episode in range(total_episodes):
        obs, infos = env.reset(seed=episode, options={'randomize': True})
        episode_reward = {agent_id: 0.0 for agent_id in env.possible_agents}
        
        for step in range(steps_per_episode):
            # Get actions from all agents
            actions = {}
            values = {}
            logps = {}
            
            for agent_id in env.possible_agents:
                action, logp, value = agents[agent_id].get_action(obs[agent_id])
                actions[agent_id] = action
                values[agent_id] = value
                logps[agent_id] = logp
            
            # Step environment
            next_obs, rewards, terms, truncs, infos = env.step(actions)
            
            # Store transitions
            for agent_id in env.possible_agents:
                agents[agent_id].store_transition(
                    obs[agent_id],
                    actions[agent_id],
                    rewards[agent_id],
                    values[agent_id],
                    logps[agent_id]
                )
                episode_reward[agent_id] += rewards[agent_id]
            
            obs = next_obs
            global_step += 1
            
            # Update agents
            if global_step % update_freq == 0:
                for agent_id in env.possible_agents:
                    _, _, last_val = agents[agent_id].get_action(obs[agent_id])
                    agents[agent_id].update(last_val=last_val)
            
            # Check termination
            if any(terms.values()) or any(truncs.values()):
                break
        
        # Store episode rewards
        for agent_id in env.possible_agents:
            episode_rewards[agent_id].append(episode_reward[agent_id])
        
        # Log progress
        avg_reward = np.mean([episode_reward[aid] for aid in env.possible_agents])
        print(f"Episode {episode + 1}/{total_episodes} | "
              f"Steps: {step + 1} | "
              f"Avg Reward: {avg_reward:.3f}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Print final statistics
    print("\nFinal Episode Rewards:")
    for agent_id in env.possible_agents:
        rewards = episode_rewards[agent_id]
        print(f"  {agent_id}: {np.mean(rewards[-10:]):.3f} (last 10 episodes)")
    
    return agents, episode_rewards


if __name__ == "__main__":
    agents, rewards = train_ppo(
        dataset="nine_intersections",
        total_episodes=5,
        update_freq=10
    )

