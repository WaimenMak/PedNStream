# -*- coding: utf-8 -*-
"""
Utility functions and wrappers for RL training on PedNet environments.

This module contains:
- RunningNormalizeWrapper: Online normalization for PettingZoo ParallelEnv
- Model save/load utilities for multi-agent systems
- Evaluation utilities
"""

import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import collections
import random

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    PPO 标准初始化技巧：
    - 使用正交初始化 (Orthogonal Initialization) 保持梯度范数。
    - 允许自定义增益 (std) 和偏置 (bias_const)。
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)

# class LazyFrameStackReplayBuffer:
#     def __init__(self, capacity, stack_size=4):
#         self.capacity = capacity
#         self.stack_size = stack_size
#         self.ptr = 0
#         self.buffer_size = 0
#         self.initialized = False
        
#         # 我们暂时不初始化数组，等第一条数据进来时再根据数据的形状初始化
#         self.states = None
#         self.actions = None
#         self.rewards = None
#         self.next_states = None
#         self.dones = None

#     def _init_memory(self, state, action, reward, next_state, done):
#         # 自动推断维度
#         state_dim = state.shape if hasattr(state, 'shape') else (len(state),)
#         action_dim = action.shape if hasattr(action, 'shape') else (len(action),)
        
#         # 预分配内存 (N, state_dim) - 注意这里只存单帧！
#         self.states = np.zeros((self.capacity, *state_dim), dtype=np.float32)
#         self.next_states = np.zeros((self.capacity, *state_dim), dtype=np.float32)
        
#         # 根据 action 是离散还是连续决定类型，这里默认 float
#         self.actions = np.zeros((self.capacity, *action_dim), dtype=np.float32)
#         self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
#         self.dones = np.zeros((self.capacity, 1), dtype=np.bool_)
        
#         self.initialized = True

#     def add(self, state, action, reward, next_state, done):
#         # 第一次调用时初始化 Buffer
#         if not self.initialized:
#             self._init_memory(state, action, reward, next_state, done)
            
#         # 存入数据
#         self.states[self.ptr] = state
#         self.actions[self.ptr] = action
#         self.rewards[self.ptr] = reward
#         self.next_states[self.ptr] = next_state
#         self.dones[self.ptr] = done

#         # 更新指针 (环形移动)
#         self.ptr = (self.ptr + 1) % self.capacity
#         self.buffer_size = min(self.buffer_size + 1, self.capacity)

#     def _get_history(self, index, source_array, current_frame=None):
#         """
#         核心逻辑：向前回溯 stack_size 帧。
#         source_array: 通常是 self.states
#         current_frame: 如果是计算 next_state 的 stack，最新的那一帧还没存进 buffer (或者存了但逻辑不同)，
#                        可以显式传入最新的帧。
#         """
#         frames = []
        
#         # 确定最新的那一帧以及其形状（用于零填充）
#         # 如果是取 state stack，最新的就是 source_array[index]
#         # 如果是取 next_state stack，最新的通常是 source_array[index] 的 next_state，即 current_frame
#         if current_frame is not None:
#             latest_frame = current_frame
#         else:
#             # 如果没有提供 current_frame，使用 source_array[index] 作为最新帧
#             latest_frame = source_array[index]
        
#         # 获取帧的形状用于零填充
#         frame_shape = latest_frame.shape if hasattr(latest_frame, 'shape') else (len(latest_frame),)
#         zero_frame = np.zeros(frame_shape, dtype=np.float32)
        
#         frames.append(latest_frame)
        
#         # 我们需要凑齐 stack_size 帧
#         # 比如 stack=4, 如果 current_frame 有值，循环 3 次；否则循环 4 次
#         needed = self.stack_size - len(frames)
        
#         for i in range(1, needed + 1):  # i从1开始，因为index本身已经在frames里了
#             # 回溯索引: index-1, index-2, ...
#             # 注意：index 是当前的 buffer 指针，我们往回找
            
#             # 修复：正确处理负数索引，避免错误的循环包装
#             prev_idx = index - i
            
#             # 边界检查关键点：
#             # 如果这一帧是 Done（上个回合结束），或者我们回溯到了 Buffer 的还没写数据的地方(buffer_size check)
#             # 就停止回溯，用零值填充剩下的位置 (Zero Padding)
            
#             # 1. 检查是否越过 buffer 有效区
#             # 如果 prev_idx < 0，说明我们回溯到了 buffer 开始之前
#             # 注意: prev_idx >= buffer_size 永远不会发生，因为 index 从 [0, buffer_size) 采样，
#             # 而 prev_idx = index - i (i >= 1)，所以 prev_idx < buffer_size 总是成立（除非 prev_idx < 0）
#             if prev_idx < 0:
#                 # Buffer 还没满，回溯到了无效区域
#                 if self.buffer_size < self.capacity:
#                     # 用零值填充（表示没有历史信息）
#                     while len(frames) < self.stack_size:
#                         frames.append(zero_frame)
#                     break
#                 else:
#                     # Buffer 已满，需要包装到 buffer 末尾
#                     # 例如：如果 index=1, i=2, 则 prev_idx=-1 -> capacity-1
#                     prev_idx = prev_idx % self.capacity
#                     # 当buffer满时，所有索引都是有效的，所以可以继续

#             # 2. 检查 Done 标记
#             # 如果 prev_idx 这一步是 Done，说明它是上一个 Episode 的结尾。
#             # 那么 prev_idx 这一帧本身是属于上一个 Episode 的，不能用！
#             # 或者是如果我们正在取 index 的 stack，遇到 index-1 是 done，说明 index 是新 Episode 的开始。
#             if self.dones[prev_idx] and len(frames) < self.stack_size:
#                 # 遇到了断点，后面的帧（其实是更早的时间）都属于上个回合
#                 # 用零值填充（表示新episode开始，没有更早的历史）
#                 while len(frames) < self.stack_size:
#                     frames.append(zero_frame)
#                 break
                
#             frames.append(source_array[prev_idx])
            
#         # 确保我们有足够的帧（如果还没填满，用零值填充）
#         if len(frames) < self.stack_size:
#             while len(frames) < self.stack_size:
#                 frames.append(zero_frame)
            
#         # 现在的 frames 是 [t, t-1, t-2, t-3]，我们需要反转成 [t-3, t-2, t-1, t]
#         return np.array(frames[::-1])

#     def sample(self, batch_size):
#         indices = np.random.randint(0, self.buffer_size, size=batch_size)
        
#         stacked_states = []
#         stacked_next_states = []
        
#         for idx in indices:
#             # 1. 构建 Current State Stack
#             # 这里的逻辑是：取 buffer[idx] 以及它之前的 k-1 帧
#             s_stack = self._get_history(idx, self.states)
#             stacked_states.append(s_stack)
            
#             # 2. 构建 Next State Stack
#             # Next State 的 stack 应该是：[..., s[t-1], s[t], s[t+1]]
#             # 其中 s[t+1] 就是存进去的 self.next_states[idx]
#             # 所以我们要基于 self.states 回溯，但是把最新的帧替换成 next_state[idx]
#             ns_stack = self._get_history(idx, self.states, current_frame=self.next_states[idx])
#             stacked_next_states.append(ns_stack)

#         # 转换格式: (Batch, Stack, Dim)
#         # 注意: 如果你用 Conv1d，通常希望 Dim 在中间，可能需要 transpose
#         return (
#             np.array(stacked_states), 
#             self.actions[indices], 
#             self.rewards[indices], 
#             np.array(stacked_next_states), 
#             self.dones[indices]
#         )

#     def size(self):
#         return self.buffer_size



# =============================================================================
# Running Normalization Wrapper
# =============================================================================

class RunningMeanStd:
    """
    Tracks running mean and variance using Welford's online algorithm.
    Same as gymnasium.wrappers.utils.RunningMeanStd but standalone.
    """
    def __init__(self, epsilon: float = 1e-4, shape: tuple = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x: np.ndarray):
        """Update running statistics with a batch of data."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count


class RunningNormalizeWrapper:
    """
    Lightweight running normalization wrapper for PettingZoo ParallelEnv.
    
    Normalizes observations and/or rewards using online mean/std estimation,
    similar to Stable Baselines 3's VecNormalize but preserving PettingZoo's
    dict-based API.
    
    Usage:
        env = PedNetParallelEnv(dataset="45_intersections", ...)
        env = RunningNormalizeWrapper(env, norm_obs=True, norm_reward=False)
        
        # Works with standard PettingZoo API
        obs, infos = env.reset()
        obs, rewards, terms, truncs, infos = env.step(actions)
    """
    
    def __init__(self, env, norm_obs: bool = True, norm_reward: bool = False,
                 clip_obs: float = 10.0, clip_reward: float = 10.0,
                 gamma: float = 0.99, training: bool = True):
        """
        Initialize the normalization wrapper.
        
        Args:
            env: PettingZoo ParallelEnv to wrap
            norm_obs: Whether to normalize observations
            norm_reward: Whether to normalize rewards
            clip_obs: Clipping range for normalized observations
            clip_reward: Clipping range for normalized rewards
            gamma: Discount factor for reward normalization (returns-based)
            training: Whether to update running statistics
        """
        self.env = env
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.training = training
        
        # Initialize running statistics for each agent
        self.obs_rms = {aid: RunningMeanStd(shape=(env.observation_space(aid).shape[0],)) 
                        for aid in env.possible_agents}
        self.ret_rms = RunningMeanStd(shape=()) if norm_reward else None
        
        # Track returns for reward normalization
        self._returns = {aid: 0.0 for aid in env.possible_agents}
    
    def __getattr__(self, name):
        """Delegate attribute access to wrapped environment."""
        return getattr(self.env, name)
    
    def reset(self, **kwargs):
        """Reset environment and normalize initial observations."""
        obs, infos = self.env.reset(**kwargs)
        
        # Reset return tracking
        self._returns = {aid: 0.0 for aid in self.env.possible_agents}
        
        if self.norm_obs:
            obs = self._normalize_obs(obs, update=self.training)
        return obs, infos
    
    def step(self, actions):
        """Step environment and normalize observations/rewards."""
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        
        if self.norm_obs:
            obs = self._normalize_obs(obs, update=self.training)
        
        if self.norm_reward:
            rewards = self._normalize_rewards(rewards, terms, update=self.training)
        
        return obs, rewards, terms, truncs, infos
    
    def _normalize_obs(self, obs: Dict[str, np.ndarray], update: bool = True) -> Dict[str, np.ndarray]:
        """Normalize observations using running mean/std, excluding gate width for gater agents."""
        normalized = {}
        for aid, o in obs.items():
            # Update RMS with all observations (for tracking statistics)
            if update:
                self.obs_rms[aid].update(o.reshape(1, -1))
            
            # Normalize the observation
            o_normalized = np.clip(
                (o - self.obs_rms[aid].mean) / np.sqrt(self.obs_rms[aid].var + 1e-8),
                -self.clip_obs, self.clip_obs
            ).astype(np.float32)
            
            # For gater agents, restore un-normalized gate width values
            agent_type = self.env.agent_manager.get_agent_type(aid)
            if agent_type == "gate":
                # Gate_width is the last feature per link - don't normalize it
                features_per_link = self.env.obs_builder.features_per_link
                num_links = len(o) // features_per_link
                # Restore original gate width values (last feature of each link)
                for i in range(num_links):
                    gate_width_idx = i * features_per_link + (features_per_link - 1)
                    o_normalized[gate_width_idx] = o[gate_width_idx]
            
            normalized[aid] = o_normalized
        
        return normalized
    
    def _normalize_rewards(self, rewards: Dict[str, float], terms: Dict[str, bool],
                          update: bool = True) -> Dict[str, float]:
        """Normalize rewards using running std of returns."""
        normalized = {}
        for aid, r in rewards.items():
            # Update return estimate
            self._returns[aid] = r + self.gamma * self._returns[aid] * (1 - float(terms[aid]))
            
            if update:
                self.ret_rms.update(np.array([self._returns[aid]]).reshape(-1, 1))
            
            # Normalize by std of returns (not mean, to preserve sign)
            normalized[aid] = np.clip(
                r / np.sqrt(self.ret_rms.var + 1e-8),
                -self.clip_reward, self.clip_reward
            )
        return normalized
    
    def set_training(self, training: bool):
        """Set training mode (whether to update running statistics)."""
        self.training = training
    
    def get_normalization_stats(self) -> Dict[str, Any]:
        """Get current normalization statistics for saving."""
        stats = {
            'obs_rms': {
                aid: {'mean': rms.mean.tolist(), 'var': rms.var.tolist(), 'count': rms.count}
                for aid, rms in self.obs_rms.items()
            }
        }
        if self.ret_rms is not None:
            stats['ret_rms'] = {
                'mean': float(self.ret_rms.mean),
                'var': float(self.ret_rms.var),
                'count': self.ret_rms.count
            }
        return stats
    
    def set_normalization_stats(self, stats: Dict[str, Any]):
        """Load normalization statistics from saved data."""
        for aid, rms_data in stats['obs_rms'].items():
            if aid in self.obs_rms:
                self.obs_rms[aid].mean = np.array(rms_data['mean'])
                self.obs_rms[aid].var = np.array(rms_data['var'])
                self.obs_rms[aid].count = rms_data['count']
        
        if 'ret_rms' in stats and self.ret_rms is not None:
            self.ret_rms.mean = stats['ret_rms']['mean']
            self.ret_rms.var = stats['ret_rms']['var']
            self.ret_rms.count = stats['ret_rms']['count']


# =============================================================================
# Model Save/Load Utilities
# =============================================================================
def save_with_best_return(agents: dict, save_dir: str, metadata: dict = None,
                          episode_returns: dict = None, best_avg_return: float = float('-inf'), global_episode: int = 0):
    """
    Save all agents' parameters to a directory.
    """
    avg_episode_return = np.mean(list(episode_returns.values()))
    
    if avg_episode_return > best_avg_return:
        best_avg_return = avg_episode_return
        
        # Save all agents with metadata about the best average return
        metadata = {
            'episode': global_episode,
            'avg_return': float(avg_episode_return),
            'best_avg_return': float(best_avg_return),
            'individual_returns': {aid: float(episode_returns[aid]) for aid in agents.keys()}
        }
        save_all_agents(agents, save_dir, metadata=metadata)
        print(f"New best average return achieved: {best_avg_return:.3f} at episode {global_episode} (saved all agents to {save_dir})")

    return best_avg_return


def save_all_agents(agents: dict, save_dir: str, metadata: dict = None,
                    normalization_stats: dict = None):
    """
    Save all agents' parameters to a directory.
    Supports both PPO (value network) and SAC (Q network) agents.
    
    Args:
        agents: Dict mapping agent_id -> PPOAgent or SACAgent
        save_dir: Directory to save models
        metadata: Optional dict with training info (episodes, dataset, etc.)
        normalization_stats: Optional normalization statistics from wrapper
    
    Structure:
        save_dir/
            checkpoint.pt   # All agents' state_dicts
            config.json     # Agent configs and metadata
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all state dicts
    checkpoint = {}
    configs = {}
    for agent_id, agent in agents.items():
        # Detect agent type: PPO has value_net, SAC has critic_1
        if hasattr(agent, 'value_net'):
            # PPO agent
            checkpoint[agent_id] = {
                'agent_type': 'PPO',
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.value_net.state_dict(),
                'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
            }
            # Store additional PPO-specific state if available
            if hasattr(agent, 'update_count'):
                checkpoint[agent_id]['update_count'] = agent.update_count
            if hasattr(agent, 'entropy_coef'):
                checkpoint[agent_id]['current_entropy_coef'] = agent.entropy_coef
        elif hasattr(agent, 'critic_1'):
            # SAC agent
            checkpoint[agent_id] = {
                'agent_type': 'SAC',
                'actor_state_dict': agent.actor.state_dict(),
                'critic_1_state_dict': agent.critic_1.state_dict(),
                'critic_2_state_dict': agent.critic_2.state_dict(),
                'target_critic_1_state_dict': agent.target_critic_1.state_dict(),
                'target_critic_2_state_dict': agent.target_critic_2.state_dict(),
                'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                'critic_1_optimizer_state_dict': agent.critic_1_optimizer.state_dict(),
                'critic_2_optimizer_state_dict': agent.critic_2_optimizer.state_dict(),
                'log_alpha_optimizer_state_dict': agent.log_alpha_optimizer.state_dict(),
                'log_alpha': agent.log_alpha.item(),
            }
        else:
            raise ValueError(f"Unknown agent type for agent {agent_id}. Expected PPO (value_net) or SAC (critic_1)")
        
        configs[agent_id] = agent.get_config()
    
    # Save checkpoint
    torch.save(checkpoint, save_path / 'checkpoint.pt')
    
    # Save configs and metadata
    config_data = {
        'agent_configs': configs,
        'metadata': metadata or {},
        'saved_at': datetime.now().isoformat(),
    }
    
    # Include normalization stats if provided
    if normalization_stats is not None:
        config_data['normalization_stats'] = normalization_stats
    
    with open(save_path / 'config.json', 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"Saved {len(agents)} agents to {save_dir}")


def load_all_agents(save_dir: str, device: str = "cpu", agent_class=None):
    """
    Load all agents from a saved directory.
    Automatically detects agent type (PPO or SAC) from saved checkpoint.
    
    Args:
        save_dir: Directory containing saved models
        device: Device to load models to
        agent_class: Optional agent class (auto-detected if not provided)
    
    Returns:
        agents: Dict mapping agent_id -> PPOAgent or SACAgent
        config_data: Full config including metadata and normalization stats
    """
    save_path = Path(save_dir)
    
    # Load configs
    with open(save_path / 'config.json', 'r') as f:
        config_data = json.load(f)
    
    # Load checkpoint
    checkpoint = torch.load(save_path / 'checkpoint.pt', map_location=device)
    
    # Recreate agents
    agents = {}
    for agent_id, config in config_data['agent_configs'].items():
        # Detect agent type from checkpoint
        agent_type = checkpoint[agent_id].get('agent_type', None)
        
        # If agent_type not in checkpoint, try to infer from config keys
        if agent_type is None:
            if 'lmbda' in config or 'epochs' in config or 'clip_eps' in config:
                agent_type = 'PPO'
            elif 'stack_size' in config or 'tau' in config or 'target_entropy' in config:
                agent_type = 'SAC'
            else:
                raise ValueError(f"Cannot determine agent type for {agent_id}")
        
        if agent_type == 'PPO':
            # Import PPOAgent
            if agent_class is None or agent_class.__name__ != 'PPOAgent':
                from rl.agents.PPO import PPOAgent
                agent_class_to_use = PPOAgent
            else:
                agent_class_to_use = agent_class
            
            # Create PPO agent with saved config
            use_stacked_obs = config.get('use_stacked_obs', False)
            
            if use_stacked_obs:
                agent = agent_class_to_use(
                    obs_dim=config['obs_dim'],
                    act_dim=config['act_dim'],
                    act_low=config['act_low'],
                    act_high=config['act_high'],
                    gamma=config['gamma'],
                    lmbda=config['lmbda'],
                    epochs=config['epochs'],
                    clip_eps=config['clip_eps'],
                    entropy_coef=config['entropy_coef'],
                    entropy_coef_decay=config.get('entropy_coef_decay', 0.995),
                    entropy_coef_min=config.get('entropy_coef_min', 0.0001),
                    use_delta_actions=config['use_delta_actions'],
                    max_delta=config['max_delta'],
                    use_stacked_obs=True,
                    stack_size=config['stack_size'],
                    hidden_size=config['hidden_size'],
                    kernel_size=config['kernel_size'],
                    device=device,
                )
            else:
                agent = agent_class_to_use(
                    obs_dim=config['obs_dim'],
                    act_dim=config['act_dim'],
                    act_low=config['act_low'],
                    act_high=config['act_high'],
                    gamma=config['gamma'],
                    lmbda=config['lmbda'],
                    epochs=config['epochs'],
                    clip_eps=config['clip_eps'],
                    entropy_coef=config['entropy_coef'],
                    entropy_coef_decay=config.get('entropy_coef_decay', 0.995),
                    entropy_coef_min=config.get('entropy_coef_min', 0.0001),
                    use_delta_actions=config['use_delta_actions'],
                    max_delta=config['max_delta'],
                    lstm_hidden_size=config['lstm_hidden_size'],
                    num_lstm_layers=config.get('num_lstm_layers', 1),
                    device=device,
                )
            
            # Load state dicts
            agent.actor.load_state_dict(checkpoint[agent_id]['actor_state_dict'])
            agent.value_net.load_state_dict(checkpoint[agent_id]['critic_state_dict'])
            agent.actor_optimizer.load_state_dict(checkpoint[agent_id]['actor_optimizer_state_dict'])
            agent.critic_optimizer.load_state_dict(checkpoint[agent_id]['critic_optimizer_state_dict'])
            
            # Restore additional PPO-specific state if available
            if 'update_count' in checkpoint[agent_id]:
                agent.update_count = checkpoint[agent_id]['update_count']
            if 'current_entropy_coef' in checkpoint[agent_id]:
                agent.entropy_coef = checkpoint[agent_id]['current_entropy_coef']
        
        elif agent_type == 'SAC':
            # Import SACAgent
            if agent_class is None or agent_class.__name__ != 'SACAgent':
                from rl.agents.SAC import SACAgent
                agent_class_to_use = SACAgent
            else:
                agent_class_to_use = agent_class
            
            # Create SAC agent with saved config
            agent = agent_class_to_use(
                obs_dim=config['obs_dim'],
                act_dim=config['act_dim'],
                act_low=config['act_low'],
                act_high=config['act_high'],
                stack_size=config['stack_size'],
                hidden_size=config['hidden_size'],
                kernel_size=config['kernel_size'],
                actor_lr=config['actor_lr'],
                critic_lr=config['critic_lr'],
                alpha_lr=config['alpha_lr'],
                target_entropy=config['target_entropy'],
                tau=config['tau'],
                gamma=config['gamma'],
                buffer_size=config.get('buffer_size', 50000),  # Default for backward compatibility
                max_delta=config['max_delta'],
                device=device,
            )
            
            # Load state dicts
            agent.actor.load_state_dict(checkpoint[agent_id]['actor_state_dict'])
            agent.critic_1.load_state_dict(checkpoint[agent_id]['critic_1_state_dict'])
            agent.critic_2.load_state_dict(checkpoint[agent_id]['critic_2_state_dict'])
            agent.target_critic_1.load_state_dict(checkpoint[agent_id]['target_critic_1_state_dict'])
            agent.target_critic_2.load_state_dict(checkpoint[agent_id]['target_critic_2_state_dict'])
            agent.actor_optimizer.load_state_dict(checkpoint[agent_id]['actor_optimizer_state_dict'])
            agent.critic_1_optimizer.load_state_dict(checkpoint[agent_id]['critic_1_optimizer_state_dict'])
            agent.critic_2_optimizer.load_state_dict(checkpoint[agent_id]['critic_2_optimizer_state_dict'])
            agent.log_alpha_optimizer.load_state_dict(checkpoint[agent_id]['log_alpha_optimizer_state_dict'])
            agent.log_alpha.data = torch.tensor(checkpoint[agent_id]['log_alpha'], dtype=torch.float).to(device)
        
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        agents[agent_id] = agent
    
    print(f"Loaded {len(agents)} agents from {save_dir}")
    return agents, config_data


def load_normalization_stats(save_dir: str) -> Optional[Dict[str, Any]]:
    """Load normalization statistics from saved config."""
    save_path = Path(save_dir)
    config_path = save_path / 'config.json'
    
    if not config_path.exists():
        return None
    
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    return config_data.get('normalization_stats')


# =============================================================================
# Evaluation Utilities  
# =============================================================================

def compute_network_throughput(network=None, simulation_dir=None):
    """
    Compute network throughput as completed demand / total demand.
    
    Network throughput measures the fraction of generated demand that successfully
    reaches destination nodes. This is a key global performance metric.
    
    Args:
        network: Network object (for live evaluation)
        simulation_dir: Path to saved simulation directory (for offline evaluation)
        
    Returns:
        dict: {
            'throughput': float,  # completed_demand / total_demand
            'completed_demand': float,  # total trips that reached destinations
            'total_demand': float,  # total trips generated at origins
            'completion_rate': float  # alias for throughput (for clarity)
        }
        
    Note:
        - For live networks: accesses node.demand and virtual_incoming_link.cumulative_inflow
        - For saved data: uses node_data.json and link_data.json
        - Virtual links for destinations are not saved, so we compute completed flow
          as sum of cumulative_outflow from real links whose end_node is a destination
    """
    import numpy as np
    from pathlib import Path
    import json

    # Offline evaluation from saved data
    sim_path = Path(simulation_dir)
    
    # Load network parameters
    network_params_path = sim_path / 'network_params.json'
    if not network_params_path.exists():
        raise FileNotFoundError(f"network_params.json not found in {simulation_dir}")
    
    with open(network_params_path, 'r') as f:
        network_params = json.load(f)
    
    origin_nodes = network_params.get('origin_nodes', [])
    destination_nodes = set(network_params.get('destination_nodes', []))
    
    # Load node data
    node_data_path = sim_path / 'node_data.json'
    if not node_data_path.exists():
        raise FileNotFoundError(f"node_data.json not found in {simulation_dir}")
    
    with open(node_data_path, 'r') as f:
        node_data = json.load(f)
    
    # Load link data
    link_data_path = sim_path / 'link_data.json'
    if not link_data_path.exists():
        raise FileNotFoundError(f"link_data.json not found in {simulation_dir}")
    
    with open(link_data_path, 'r') as f:
        link_data = json.load(f)
    
    # Sum total demand from origin nodes
    total_demand = 0.0
    for origin_id in origin_nodes:
        origin_str = str(origin_id)
        if origin_str in node_data:
            demand_array = node_data[origin_str].get('demand', [])
            if demand_array:
                total_demand += sum(demand_array)
    
    # Compute completed flow: sum cumulative_outflow from links ending at destination nodes
    # Link keys in link_data are in format "u-v" where v is the end node
    completed_demand = 0.0
    processed_links = set()  # Avoid double counting if link appears multiple times
    
    for link_key, link_info in link_data.items():
        if link_key in processed_links:
            continue
        
        # Parse link key "u-v" to get end node
        try:
            parts = link_key.split('-')
            if len(parts) == 2:
                start_node, end_node = int(parts[0]), int(parts[1])
                
                # Check if this link ends at a destination node
                if end_node in destination_nodes:
                    cum_outflow = link_info.get('cumulative_outflow', [])
                    if cum_outflow:
                        # Add the final cumulative outflow (last timestep)
                        completed_demand += cum_outflow[-1]
                        processed_links.add(link_key)
        except (ValueError, IndexError):
            # Skip malformed link keys
            continue
        
    
    # Compute throughput
    if total_demand > 0:
        throughput = completed_demand / total_demand
    else:
        throughput = 0.0
    
    return {
        'throughput': throughput,
        'completed_demand': completed_demand,
        'total_demand': total_demand,
        'completion_rate': throughput  # Alias for clarity
    }


def compute_network_travel_time(simulation_dir=None):
    """
    Compute average travel time using flow-weighted approach.
    
    This metric measures the average time pedestrians spend traveling through the network,
    weighted by the number of people exiting each link at each timestep. This captures
    congestion effects: when links are congested, travel_time increases, and this metric
    reflects that directly.
    
    Formula: 
        TotalTravelTime = sum_{links, t} travel_time[link, t] * outflow[link, t]
        AvgTravelTime = TotalTravelTime / TotalCompletedFlow
    
    Args:
        simulation_dir: Path to saved simulation directory
        
    Returns:
        dict: {
            'avg_travel_time': float,  # Average travel time per person (seconds)
            'total_travel_time': float,  # Total person-seconds of travel time
            'total_flow_weight': float,  # Total outflow (for normalization)
            'flow_weighted_tt': float  # Alias for avg_travel_time
        }
        
    Note:
        - Uses outflow-weighted travel time to capture actual experience of people exiting
        - Accounts for congestion: when density is high, travel_time increases automatically
        - More meaningful than simple "time in system" as it's normalized by link capacity
    """
    import numpy as np
    from pathlib import Path
    import json
    
    sim_path = Path(simulation_dir)
    
    # Load link data
    link_data_path = sim_path / 'link_data.json'
    if not link_data_path.exists():
        raise FileNotFoundError(f"link_data.json not found in {simulation_dir}")
    
    with open(link_data_path, 'r') as f:
        link_data = json.load(f)
    
    # Load network parameters for unit_time
    network_params_path = sim_path / 'network_params.json'
    unit_time = 1.0  # Default
    if network_params_path.exists():
        with open(network_params_path, 'r') as f:
            network_params = json.load(f)
            unit_time = network_params.get('unit_time', 1.0)
    
    total_travel_time = 0.0  # Person-seconds of travel time
    total_flow_weight = 0.0  # Total outflow (for normalization)
    
    for link_key, link_info in link_data.items():
        travel_time_array = link_info.get('travel_time', [])
        outflow_array = link_info.get('outflow', [])
        
        if not travel_time_array or not outflow_array:
            continue
        
        # Ensure arrays have same length
        min_len = min(len(travel_time_array), len(outflow_array))
        travel_time_array = travel_time_array[:min_len]
        outflow_array = outflow_array[:min_len]
        
        # Sum travel_time[t] * outflow[t] over all timesteps
        # This gives "person-seconds of travel time" for people exiting this link
        for t in range(min_len):
            tt = travel_time_array[t]
            flow = outflow_array[t]
            
            # Skip invalid values
            if tt is None or flow is None or tt < 0 or flow < 0:
                continue
            
            # Travel time is in seconds, outflow is in pedestrians per timestep
            total_travel_time += tt * flow
            total_flow_weight += flow
    
    # Compute average travel time
    if total_flow_weight > 0:
        avg_travel_time = total_travel_time / total_flow_weight
    else:
        avg_travel_time = 0.0
    
    return {
        'avg_travel_time': avg_travel_time,
        'total_travel_time': total_travel_time,
        'total_flow_weight': total_flow_weight,
        'flow_weighted_tt': avg_travel_time  # Alias
    }


def compute_network_congestion_metric(simulation_dir=None, threshold_ratio=0.7):
    """
    Compute congestion metric using density-normalized approach.
    
    This metric measures the severity and duration of congestion, accounting for
    link capacity (k_jam). A link with high density relative to its jam density
    is considered congested, regardless of absolute pedestrian count.
    
    Formula:
        NormalizedDensity[t] = density[t] / k_jam
        CongestionTime = sum_{links, t} max(0, NormalizedDensity[t] - threshold) * area * dt
        
    Args:
        simulation_dir: Path to saved simulation directory
        threshold_ratio: Threshold for congestion (default 0.7 = 70% of jam density)
        
    Returns:
        dict: {
            'congestion_time': float,  # Total congestion-seconds (area-time weighted)
            'avg_congestion_density': float,  # Average normalized density above threshold
            'congestion_fraction': float,  # Fraction of network-time that is congested
            'total_area_time': float  # Total area-time for normalization
        }
    """
    import numpy as np
    from pathlib import Path
    import json
    
    sim_path = Path(simulation_dir)
    
    # Load link data
    link_data_path = sim_path / 'link_data.json'
    if not link_data_path.exists():
        raise FileNotFoundError(f"link_data.json not found in {simulation_dir}")
    
    with open(link_data_path, 'r') as f:
        link_data = json.load(f)
    
    # Load network parameters for unit_time
    network_params_path = sim_path / 'network_params.json'
    unit_time = 1.0  # Default
    if network_params_path.exists():
        with open(network_params_path, 'r') as f:
            network_params = json.load(f)
            unit_time = network_params.get('unit_time', 1.0)
    
    total_congestion_time = 0.0  # Area-time weighted congestion
    total_area_time = 0.0  # Total area-time for normalization
    congestion_timesteps = 0
    total_timesteps = 0
    
    for link_key, link_info in link_data.items():
        density_array = link_info.get('density', [])
        params = link_info.get('parameters', {})
        k_jam = params.get('k_jam', 1.0)
        length = params.get('length', 1.0)
        width = params.get('width', 1.0)
        area = length * width
        
        if not density_array or k_jam <= 0:
            continue
        
        for t, density in enumerate(density_array):
            if density is None or density < 0:
                continue
            
            # Normalize density by jam density
            normalized_density = density / k_jam
            
            # Area-time for this link-timestep
            area_time = area * unit_time
            total_area_time += area_time
            total_timesteps += 1
            
            # Check if congested (above threshold)
            if normalized_density > threshold_ratio:
                congestion_timesteps += 1
                # Weight by excess density above threshold
                excess_density = normalized_density - threshold_ratio
                total_congestion_time += excess_density * area_time
    
    # Compute metrics
    if total_area_time > 0:
        avg_congestion_density = total_congestion_time / total_area_time
        congestion_fraction = congestion_timesteps / total_timesteps if total_timesteps > 0 else 0.0
    else:
        avg_congestion_density = 0.0
        congestion_fraction = 0.0
    
    return {
        'congestion_time': total_congestion_time,
        'avg_congestion_density': avg_congestion_density,
        'congestion_fraction': congestion_fraction,
        'total_area_time': total_area_time
    }


def evaluate_agents(env, agents, delta_actions: bool = False, deterministic: bool = True,
                    seed: int = None, no_control: bool = False, randomize: bool = False, 
                    save_dir: str = None, verbose: bool = True):
    """
    Evaluate trained agents on a specific environment setting without training.
    Useful for comparing with rule-based agents.
    
    Args:
        env: PettingZoo ParallelEnv (optionally wrapped with RunningNormalizeWrapper)
        agents: Dict mapping agent_id -> PPOAgent (or any agent with take_action method)
        delta_actions: If True, agents output delta actions
        deterministic: If True, use deterministic actions (mean, no sampling)
        seed: Random seed for environment reset
        no_control: If True, skip action computation (no control baseline)
        randomize: If True, randomize environment at reset
        save_dir: If provided, save simulation results to this directory
        verbose: Whether to print results
        
    Returns:
        dict: {
            'episode_rewards': {agent_id: total_reward},
            'avg_reward': float,
            'total_reward': float
        }
    """
    # Set wrapper to evaluation mode if applicable
    if hasattr(env, 'set_training'):
        env.set_training(False)
    
    # Reset environment with specified settings
    reset_options = {'randomize': randomize} if randomize else None
    obs, infos = env.reset(seed=seed, options=reset_options)
    
    # Initialize state history queues for stacked observations (for agents that need them)
    state_history_queue = {}
    state_stack = {}
    for agent_id, agent in agents.items():
        if hasattr(agent, 'stack_size'):
            stack_size = agent.stack_size
            state_history_queue[agent_id] = collections.deque(maxlen=stack_size)
            # Initialize queue with first observation repeated
            for _ in range(stack_size):
                state_history_queue[agent_id].append(obs[agent_id])
            state_stack[agent_id] = np.array(state_history_queue[agent_id])
    
    episode_rewards = {agent_id: 0.0 for agent_id in agents.keys()}
    done = False
    step = 0
    
    if verbose:
        print(f"Evaluating agents for {env.simulation_steps} steps...")
    
    while not done:
        actions = {}
        absolute_actions = {}
        
        if no_control:
            # No control baseline - skip action computation
            pass
        else:
            # Get actions from all agents
            for agent_id, agent in agents.items():
                # Use stacked state if agent uses stacked observations, otherwise use single observation
                if agent_id in state_stack:
                    agent_state = state_stack[agent_id]
                else:
                    agent_state = obs[agent_id]
                
                # Check if agent has deterministic option (PPOAgent)
                if hasattr(agent, 'take_action'):
                    try:
                        action = agent.take_action(agent_state, deterministic=deterministic)
                    except TypeError:
                        # Agent doesn't support deterministic kwarg
                        action = agent.take_action(agent_state)
                else:
                    action = agent.act(agent_state)
                
                if delta_actions and hasattr(agent, 'act_low'):
                    # Convert delta to absolute action
                    absolute_action = obs[agent_id].reshape(agent.act_dim, -1)[:, -1] + action
                    absolute_action = np.clip(absolute_action, agent.act_low, agent.act_high)
                    absolute_actions[agent_id] = absolute_action
                else:
                    absolute_actions[agent_id] = action
                actions[agent_id] = action
        
        # Step environment
        # print(f"Step {step}: Actions: {absolute_actions}")
        next_obs, rewards, terms, truncs, infos = env.step(absolute_actions)
        
        # Update state history queues for stacked observations
        for agent_id in state_history_queue.keys():
            state_history_queue[agent_id].append(next_obs[agent_id])
            state_stack[agent_id] = np.array(state_history_queue[agent_id])
        
        # Accumulate rewards
        for agent_id in agents.keys():
            episode_rewards[agent_id] += rewards[agent_id]
        
        obs = next_obs
        step += 1
        
        # Check if episode is done
        done = any(terms.values()) or any(truncs.values())
    
    # Calculate summary metrics
    total_reward = sum(episode_rewards.values())
    avg_reward = np.mean(list(episode_rewards.values()))
    
    # Print results
    if verbose:
        print("=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        for agent_id in agents.keys():
            print(f"  Agent {agent_id}: {episode_rewards[agent_id]:.3f}")
        print(f"  Average reward: {avg_reward:.3f}")
        print(f"  Total reward: {total_reward:.3f}")
        print("=" * 60)
    
    # Save simulation if requested
    if save_dir is not None:
        env.save(save_dir)
        if verbose:
            print(f"Saved simulation to {save_dir}")
    
    return {
        'episode_rewards': episode_rewards,
        'avg_reward': avg_reward,
        'total_reward': total_reward
    }


# =============================================================================
# GAE Advantage Computation
# =============================================================================

def compute_gae(gamma: float, lmbda: float, td_delta: torch.Tensor) -> torch.Tensor:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        gamma: Discount factor
        lmbda: GAE lambda parameter
        td_delta: TD errors array
        
    Returns:
        advantages: GAE advantages array
    """
    advantage_list = []
    advantage = 0.0
    td_delta = td_delta.numpy()
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)

