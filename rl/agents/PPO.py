# -*- coding: utf-8 -*-
# @Time    : 05/01/2026 15:06
# @Author  : mmai
# @FileName: PPO.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from rl.rl_utils import compute_gae, save_with_best_return, layer_init
import math
import os
import collections
from .SAC import MLPEncoder, StackedEncoder
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch

class LSTMPolicyNetwork(nn.Module):
    """Stateful LSTM-based policy network that maintains hidden state across timesteps."""
    def __init__(self, obs_dim, act_dim, hidden_size=64, num_layers=1,
                 min_std=1e-3, max_std=10.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.min_std = min_std
        self.max_std = max_std

        # LSTM layer: input is (batch, seq_len=1, obs_dim) for single timestep
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        num_gates = 4
        gate_hidden_size = 32
        # Output heads
        # self.gate_width_head = nn.Linear(num_gates, gate_hidden_size)
        self.mean_head = nn.Linear(hidden_size, act_dim)
        self.std_head = nn.Linear(hidden_size, act_dim)
        # self.mean_head = nn.Linear(hidden_size + gate_hidden_size, act_dim)
        # self.std_head = nn.Linear(hidden_size + gate_hidden_size, act_dim)

    def forward(self, x, hidden=None):
        """
        Forward pass with optional hidden state.

        Args:
            x: Single observation of shape (batch, obs_dim)
            hidden: Optional tuple (h, c) of hidden states

        Returns:
            mean: Action mean
            std: Action std
            hidden: Updated hidden state tuple (h, c)
        """
        # Add sequence dimension: (batch, Steps, obs_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # gate_widths = x[0, :, :].reshape(x.shape[1], self.act_dim, -1)[:,:,-1]
        # latent_gate_widths = self.gate_width_head(gate_widths)
        # LSTM forward pass
        lstm_out, hidden_out = self.lstm(x, hidden)  # lstm_out: (batch, 1, hidden_size)

        # Extract features from output
        features = lstm_out.squeeze(0)  # (1, steps, hidden_size) --> (steps, hidden_size)
        # features = torch.cat([features, latent_gate_widths], dim=1)

        # Compute mean and std
        mean = self.mean_head(F.relu(features))
        std = F.softplus(self.std_head(F.relu(features))).clamp(self.min_std, self.max_std)

        return mean, std, hidden_out


class LSTMValueNetwork(nn.Module):
    """Stateful LSTM-based value network that maintains hidden state across timesteps."""
    def __init__(self, obs_dim, hidden_size=64, num_layers=1):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = 4
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Layer normalization for stabilizing training
        # self.layer_norm = nn.LayerNorm(hidden_size)
        num_gates = self.act_dim
        # gate_hidden_size = 32
        # self.gate_width_head = nn.Linear(num_gates, gate_hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)
        # self.value_head = nn.Linear(hidden_size + gate_hidden_size, 1)

    def forward(self, x, hidden=None):
        """
        Forward pass with optional hidden state.

        Args:
            x: Single observation of shape (batch, obs_dim) or sequence (batch, seq_len, obs_dim)
            hidden: Optional tuple (h, c) of hidden states

        Returns:
            value: State value estimate
            hidden: Updated hidden state tuple (h, c)
        """
        # Add sequence dimension if single observation
        if x.dim() == 2:
            x = x.unsqueeze(1)

        lstm_out, hidden_out = self.lstm(x, hidden)
        features = lstm_out[:, -1, :]  # Take last timestep
        # features = self.layer_norm(features)  # Apply layer normalization
        value = self.value_head(F.relu(features))

        return value, hidden_out

    def forward_all_steps(self, x, hidden=None):
        lstm_out, hidden_out = self.lstm(x, hidden)
        features = lstm_out  # (batch, seq_len, hidden_size)
        # features = self.layer_norm(features)  # Apply layer normalization
        # values = self.value_head(F.relu(features))  # (batch, seq_len, 1)
        
        # gate_widths = x[0, :, :].reshape(x.shape[1], self.act_dim, -1)[:,:,-1]
        # latent_gate_widths = self.gate_width_head(gate_widths)
        # features = torch.cat([features.squeeze(0), latent_gate_widths], dim=1)
        values = self.value_head(F.relu(features))
        return values, hidden_out


# =============================================================================
# Stacked State Networks (similar to SAC)
# =============================================================================

class StackedPolicyNetwork(nn.Module):
    """Policy network for stacked observations."""
    def __init__(self, obs_dim, act_dim, stack_size=4, hidden_size=64, kernel_size=3,
                 min_std=1e-3, max_std=10.0):
        super(StackedPolicyNetwork, self).__init__()
        # self.encoder = StackedEncoder(obs_dim, stack_size=stack_size,
        #                              hidden_size=hidden_size, kernel_size=kernel_size)
        self.encoder = MLPEncoder(obs_dim, stack_size=stack_size, hidden_size=hidden_size)
        self.fc = layer_init(nn.Linear(self.encoder.ouput_dim, hidden_size), std=np.sqrt(2))
        self.fc_mu = layer_init(nn.Linear(hidden_size, act_dim), std=0.01)
        self.fc_std = layer_init(nn.Linear(hidden_size, act_dim), std=0.01)
        self.min_std = min_std
        self.max_std = max_std
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # x: (batch, seq, feat) or (batch, feat) if single observation
        if x.dim() == 2:
            # Single observation - add sequence dimension
            x = x.unsqueeze(1)
        if x.dim() == 3:
            x = x.transpose(1, 2)  # (batch, feat, seq)
        zs = self.encoder(x)
        features = self.fc(zs)
        features = self.ln(features)
        mu = self.fc_mu(F.relu(features))
        std = F.softplus(self.fc_std(F.relu(features))).clamp(self.min_std, self.max_std)
        return mu, std


class StackedValueNetwork(nn.Module):
    """Value network for stacked observations."""
    def __init__(self, obs_dim, stack_size=4, hidden_size=64, kernel_size=3):
        super(StackedValueNetwork, self).__init__()
        # self.encoder = StackedEncoder(obs_dim, stack_size=stack_size,
        #                              hidden_size=hidden_size, kernel_size=kernel_size)
        self.encoder = MLPEncoder(obs_dim, stack_size=stack_size, hidden_size=hidden_size)
        self.fc = layer_init(nn.Linear(self.encoder.ouput_dim, hidden_size), std=np.sqrt(2))
        self.value_head = layer_init(nn.Linear(hidden_size, 1), std=1.0)
        # self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # x: (batch, seq, feat) or (batch, feat) if single observation
        if x.dim() == 2:
            # Single observation - add sequence dimension
            x = x.unsqueeze(1)
        if x.dim() == 3:
            x = x.transpose(1, 2)  # (batch, feat, seq)
        zs = self.encoder(x)
        features = self.fc(zs)
        # features = self.ln(features)
        value = self.value_head(F.relu(features))
        return value



def train_on_policy_multi_agent(env, agents, delta_actions=False, num_episodes=50,
                                randomize=False, seed=None,
                                agents_saved_dir: str = None):
    """
    Train multiple on-policy agents (PPO) in a multi-agent environment.

    Args:
        env: PettingZoo ParallelEnv
        agents: Dict mapping agent_id -> PPOAgent
        delta_actions: If True, agents output delta actions
        num_episodes: Total number of episodes to train
        randomize: If True, randomize environment at reset
        save_freq: Save simulation results every N episodes (0 = disabled)
        save_dir: Base directory for saving simulation results

    Returns:
        return_dict: Dict mapping agent_id -> list of episode returns
    """
    # Initialize return tracking for each agent
    return_dict = {agent_id: [] for agent_id in agents.keys()}
    global_episode = 0  # Track global episode count for saving
    
    # Track best average return across all agents (initialize to negative infinity)
    best_avg_return = float('-inf')
    
    # Check if any agent uses stacked observations
    first_agent_id = next(iter(agents))
    uses_stacked_obs = hasattr(agents[first_agent_id], 'stack_size')
    
    # Initialize state history queues for stacked observations
    if uses_stacked_obs:
    # history queue for states stack
        first_agent_id = next(iter(agents))
        stack_size = agents[first_agent_id].stack_size
        state_history_queue = {agent_id: collections.deque(maxlen=stack_size) for agent_id in agents.keys()}

    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                # Reset buffers for all agents at start of episode
                for agent in agents.values():
                    agent.reset_buffer()

                # Reset environment
                # if i_episode % 10 == 0:
                #     obs, infos = env.reset(seed=42)
                # else:
                obs, infos = env.reset(seed=seed, options={'randomize': randomize})
                
                # Initialize state history queues for stacked observations
                state_stack = {}
                if uses_stacked_obs:
                    for agent_id in state_history_queue.keys():
                        state_history_queue[agent_id].clear()
                        for _ in range(agents[agent_id].stack_size):
                            state_history_queue[agent_id].append(obs[agent_id])
                        state_stack[agent_id] = np.array(state_history_queue[agent_id])
                
                episode_returns = {agent_id: 0.0 for agent_id in agents.keys()}
                done = False
                step = 0 # only for progress bar

                # Exploration phase
                while not done:
                    # Collect actions from all agents
                    actions = {}
                    absolute_actions = {}
                    # Make actions for all agents
                    # if step == 230:
                    #     pass
                    for agent_id, agent in agents.items():
                        # Use stacked state if agent uses stacked observations, otherwise use single observation
                        if agent_id in state_stack:
                            agent_state = state_stack[agent_id]
                        else:
                            agent_state = obs[agent_id]
                        action = agent.take_action(agent_state)
                        if delta_actions:
                            absolute_action = obs[agent_id].reshape(agents[agent_id].act_dim, -1)[:,-1] + action
                            absolute_action = np.clip(absolute_action, agents[agent_id].act_low, agents[agent_id].act_high)
                            absolute_actions[agent_id] = absolute_action
                        else:
                            absolute_actions[agent_id] = action
                        actions[agent_id] = action
                    # Step environment with all actions
                    next_obs, rewards, terms, truncs, infos = env.step(absolute_actions)
                    next_state_stack = {}
                    
                    # Store transitions for all agents (before updating state queues)
                    for agent_id, agent in agents.items():
                        # Get the state that was used for action selection
                        if agent_id in state_stack:
                            # stored_state = state_stack[agent_id].copy()  # Current state (before update)
                            # For next_state, we'll create the updated stack
                            # Build next state stack: current stack without oldest, plus new observation
                            state_history_queue[agent_id].append(next_obs[agent_id])
                            next_state_stack[agent_id] = np.array(state_history_queue[agent_id])
                            stored_state = state_stack[agent_id]
                            stored_next_state = next_state_stack[agent_id]

                        else:
                            stored_state = obs[agent_id]
                            stored_next_state = next_obs[agent_id]
                        
                        agent.store_transition(
                            state = stored_state,
                            action = actions[agent_id],
                            next_state = stored_next_state,
                            reward = rewards[agent_id],
                            done = terms[agent_id],
                        )
                        episode_returns[agent_id] += rewards[agent_id]
                    

                    obs = next_obs
                    # Update state history queues for stacked observations (after storing transitions)
                    if uses_stacked_obs:
                        state_stack = next_state_stack
                    

                    step += 1

                    # Check if episode is done
                    done = any(terms.values()) or any(truncs.values())

                # Store episode returns for all agents
                for agent_id in agents.keys():
                    return_dict[agent_id].append(episode_returns[agent_id])

                # Update all agents
                for agent_id, agent in agents.items():
                    agent.update()

                # Increment global episode counter
                global_episode += 1

                # Save all agents when average return across agents achieves new best
                if agents_saved_dir and global_episode > num_episodes/2: # save after half of the training
                    best_avg_return = save_with_best_return(agents, agents_saved_dir, episode_returns=episode_returns, best_avg_return=best_avg_return, global_episode=global_episode)                  
                # Update progress bar
                if (i_episode+1) % 10 == 0:
                    avg_return = np.mean([np.mean(return_dict[aid][-10:]) for aid in agents.keys()])
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes/10 * i + i_episode+1),
                        'avg_return': '%.3f' % avg_return,
                        'steps': step
                    })
                pbar.update(1)
                # print episode rewards of all agents
                for agent_id in agents.keys():
                    print(f"Agent {agent_id} episode reward: {episode_returns[agent_id]}")
                print(f"All agents episode reward: {sum(episode_returns.values())}")

    return return_dict, episode_returns

# Use compute_gae from rl_utils instead


class PPOAgent:
    """PPO implementation for continuous action spaces with stateful LSTM policy."""

    def __init__(self, obs_dim, act_dim, act_low, act_high, actor_lr=3e-4, critic_lr=6e-4,
                 gamma=0.99, lmbda=0.95, epochs=10, device="cpu",
                 clip_eps=0.2, entropy_coef=0.01, entropy_coef_decay=0.995, 
                 entropy_coef_min=0, kl_tolerance=0.01,
                 use_delta_actions=False, max_delta=2.5,
                 lstm_hidden_size=64, num_lstm_layers=1,
                 use_stacked_obs=False, stack_size=4, hidden_size=64, kernel_size=3,
                 use_gat_lstm=False, gat_hidden_size=64, gat_num_heads=4):
        """
        Initialize PPO agent with LSTM, stacked observation, or GAT-LSTM networks.

        Args:
            obs_dim: Observation space dimension
            act_dim: Action space dimension
            act_low: Lower bound of action space
            act_high: Upper bound of action space
            actor_lr: Learning rate for actor
            critic_lr: Learning rate for critic
            gamma: Discount factor
            lmbda: GAE lambda parameter
            epochs: Number of epochs for PPO update
            clip_eps: PPO clipping parameter
            entropy_coef: Initial entropy bonus coefficient
            entropy_coef_decay: Exponential decay factor per update (default 0.995)
            entropy_coef_min: Minimum entropy coefficient (default 0.0001)
            kl_tolerance: KL divergence tolerance for early stopping
            use_delta_actions: If True, agent outputs delta actions
            max_delta: Maximum delta per step (only used if use_delta_actions=True)
            lstm_hidden_size: Hidden size for LSTM layers
            num_lstm_layers: Number of LSTM layers
            use_stacked_obs: If True, use stacked observation networks (similar to SAC)
            stack_size: Number of observations to stack (only used if use_stacked_obs=True)
            hidden_size: Hidden size for stacked networks (only used if use_stacked_obs=True)
            kernel_size: Kernel size for Conv1d in stacked encoder (only used if use_stacked_obs=True)
            use_gat_lstm: If True, use GAT-LSTM architecture (Temporal -> Spatial)
            features_per_link: Features per link for GAT-LSTM (required if use_gat_lstm=True)
            gat_hidden_size: Hidden size for GAT layer (only used if use_gat_lstm=True)
            gat_num_heads: Number of attention heads in GAT (only used if use_gat_lstm=True)
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_low = torch.tensor(act_low, dtype=torch.float32)
        self.act_high = torch.tensor(act_high, dtype=torch.float32)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.clip_eps = clip_eps
        
        # Entropy coefficient with exponential decay
        self.entropy_coef_initial = entropy_coef
        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay
        self.entropy_coef_min = entropy_coef_min
        self.update_count = 0  # Track number of updates for decay
        
        self.kl_tolerance = kl_tolerance
        self.transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}

        # Delta action settings
        self.use_delta_actions = use_delta_actions
        self.max_delta = max_delta

        # Network type settings
        self.use_stacked_obs = use_stacked_obs
        
        if use_stacked_obs:
            # Stacked observation settings
            self.stack_size = stack_size
            self.hidden_size = hidden_size
            self.kernel_size = kernel_size
            # Create stacked networks
            self.actor = StackedPolicyNetwork(obs_dim, act_dim, stack_size=stack_size,
                                            hidden_size=hidden_size, kernel_size=kernel_size)
            self.value_net = StackedValueNetwork(obs_dim, stack_size=stack_size,
                                               hidden_size=hidden_size, kernel_size=kernel_size)
        else:
            # LSTM settings
            self.lstm_hidden_size = lstm_hidden_size
            self.num_lstm_layers = num_lstm_layers
            # Hidden states for actor and critic (reset at episode start)
            self.actor_hidden = None
            self.critic_hidden = None
            # Create LSTM networks
            self.actor = LSTMPolicyNetwork(obs_dim, act_dim, hidden_size=lstm_hidden_size,
                                          num_layers=num_lstm_layers)
            self.value_net = LSTMValueNetwork(obs_dim, hidden_size=lstm_hidden_size,
                                             num_layers=num_lstm_layers)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=critic_lr)
        self.device = device

    def reset_buffer(self):
        """Clear rollout buffer and reset LSTM hidden states (if using LSTM or GAT-LSTM)."""
        self.transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        # Reset LSTM hidden states (only if using LSTM or GAT-LSTM, will be initialized in first forward pass)
        if not self.use_stacked_obs:
            self.actor_hidden = None
            self.critic_hidden = None

    def store_transition(self, state, action, next_state, reward, done):
        """Store transition in buffer."""
        self.transition_dict['states'].append(state)
        self.transition_dict['actions'].append(action)
        self.transition_dict['next_states'].append(next_state)
        self.transition_dict['rewards'].append(reward)
        self.transition_dict['dones'].append(done)

    def take_action(self, state, deterministic: bool = False):
        """
        Take action given state using LSTM, stacked, or GAT-LSTM networks.

        Args:
            state: Observation array (single observation) or stacked observations (stack_size, obs_dim)
            deterministic: If True, use mean action (no sampling). Useful for evaluation.

        Returns:
            action: Action array of shape (act_dim,)
        """
        # Convert state to tensor
        state_array = np.array(state)
        if self.use_stacked_obs:
            # Stacked observations: (stack_size, obs_dim) -> (1, stack_size, obs_dim)
            if state_array.ndim == 2:
                state_tensor = torch.tensor(state_array, dtype=torch.float).unsqueeze(0).to(self.device)
            else:
                state_tensor = torch.tensor(state_array, dtype=torch.float).to(self.device)
                if state_tensor.dim() == 2:
                    state_tensor = state_tensor.unsqueeze(0)
        else:
            # Single observation: (obs_dim,) -> (1, obs_dim)
            state_tensor = torch.tensor(state_array, dtype=torch.float).unsqueeze(0).to(self.device)

        # Forward pass through actor
        if self.use_stacked_obs:
            mu, sigma = self.actor(state_tensor)
        else:
            mu, sigma, self.actor_hidden = self.actor(state_tensor, self.actor_hidden)

        if deterministic:
            # Use mean action for evaluation
            action = mu
        else:
            # Sample from distribution for exploration during training
            action_dist = torch.distributions.Normal(mu, sigma)
            action = action_dist.sample()

        if self.use_delta_actions:
            # Agent outputs delta in [-max_delta, +max_delta]
            delta = torch.clamp(action, -self.max_delta, self.max_delta)
            return delta.cpu().detach().numpy().squeeze()
        else:
            # Direct absolute action
            action = torch.clamp(action, self.act_low, self.act_high)
            return action.cpu().detach().numpy().squeeze()

    def update(self):
        """Update policy and value networks using collected trajectory."""
        # Convert trajectory to tensors
        # For stacked obs: states will be (T, stack_size, obs_dim)
        # For LSTM: states will be (1, T, obs_dim)
        states = torch.tensor(np.array(self.transition_dict['states']),
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(self.transition_dict['actions'])).view(-1, self.act_dim).to(
            self.device)  # (T, act_dim)
        rewards = torch.tensor(np.array(self.transition_dict['rewards']),
                               dtype=torch.float).view(-1, 1).to(self.device)  # (T, 1)
        next_states = torch.tensor(np.array(self.transition_dict['next_states']),
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(self.transition_dict['dones']),
                             dtype=torch.float).view(-1, 1).to(self.device)  # (T, 1)

        # Prepare sequences for network forward pass
        if self.use_stacked_obs:
            pass
        else:
            # LSTM: states is (T, obs_dim)
            # Add batch dimension: (1, T, obs_dim)
            states_seq = states.unsqueeze(0)  # (1, T, obs_dim)
            next_states_seq = next_states.unsqueeze(0)  # (1, T, obs_dim)

        # Compute targets with no_grad - these are constants for the update
        with torch.no_grad():
            # Process entire sequences through value network
            if self.use_stacked_obs:
                # For stacked, we need to process each timestep separately or reshape
                # StackedValueNetwork.forward_all_steps expects (batch, seq, stack_size, obs_dim)
                # But we have (1, T, stack_size, obs_dim), which is correct
                next_values = self.value_net(next_states)
                current_values = self.value_net(states)
            else:
                # LSTM version
                next_values, _ = self.value_net.forward_all_steps(next_states_seq)  # (1, T, 1)
                next_values = next_values.squeeze(0)  # (T, 1)

                current_values, _ = self.value_net.forward_all_steps(states_seq)  # (1, T, 1)
                current_values = current_values.squeeze(0)  # (T, 1)

            td_target = rewards + self.gamma * next_values * (1 - dones)
            td_delta = td_target - current_values

            # Use shared compute_gae function
            advantage = compute_gae(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

            # Normalize advantage (crucial for stable training)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            # Process sequence through actor to get old log probs
            if self.use_stacked_obs:
                mu, std = self.actor(states)
            else:
                mu, std, _ = self.actor(states_seq)  # (1, T, act_dim)
                mu = mu.squeeze(0)  # (T, act_dim)
                std = std.squeeze(0)  # (T, act_dim)
            action_dist = torch.distributions.Normal(mu, std)
            old_log_probs = action_dist.log_prob(actions)

        # PPO update epochs
        for _ in range(self.epochs):
            # Forward pass through actor
            if self.use_stacked_obs:
                mu, std = self.actor(states)
            else:
                mu, std, _ = self.actor(states_seq)
                mu = mu.squeeze(0)
                std = std.squeeze(0)
            action_dist = torch.distributions.Normal(mu, std)
            log_probs = action_dist.log_prob(actions)

            # Clamp log_prob difference to prevent ratio explosion
            log_ratio = (log_probs - old_log_probs).clamp(-20, 20)
            ratio = torch.exp(log_ratio)

            # PPO clipped loss
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                1 + self.clip_eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))

            # Entropy loss, use gaussian entropy formula
            # entropy = 0.5 * torch.log(2 * math.pi * math.e * std**2)
            # entropy_loss = -torch.mean(entropy)
            # actor_loss += self.entropy_coef * entropy_loss

            # Value loss
            if self.use_stacked_obs:
                current_values = self.value_net(states)
            else:
                current_values, _ = self.value_net.forward_all_steps(states_seq)
                current_values = current_values.squeeze(0)
            critic_loss = torch.mean(F.mse_loss(current_values, td_target))

            # Backprop and update
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            # Store critic loss for later plotting (append after .item() conversion)
            if not hasattr(self, "critic_loss_history"):
                self.critic_loss_history = []
            self.critic_loss_history.append(critic_loss.item())

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)

            self.actor_optimizer.step()
            self.critic_optimizer.step()
        
        # # --- Configuration ---
        # bptt_batch_size = 128  # Size of the truncated chunk
        # # ---------------------
        #
        # # Convert trajectory to tensors
        # states = torch.tensor(np.array(self.transition_dict['states']),
        #                       dtype=torch.float).to(self.device)  # (T, obs_dim)
        # actions = torch.tensor(np.array(self.transition_dict['actions'])).view(-1, self.act_dim).to(
        #     self.device)  # (T, act_dim)
        # rewards = torch.tensor(np.array(self.transition_dict['rewards']),
        #                        dtype=torch.float).view(-1, 1).to(self.device)  # (T, 1)
        # next_states = torch.tensor(np.array(self.transition_dict['next_states']),
        #                            dtype=torch.float).to(self.device)  # (T, obs_dim)
        # dones = torch.tensor(np.array(self.transition_dict['dones']),
        #                      dtype=torch.float).view(-1, 1).to(self.device)  # (T, 1)
        #
        # # Add batch dimension for LSTM: (1, T, obs_dim)
        # states_seq = states.unsqueeze(0)
        # next_states_seq = next_states.unsqueeze(0)
        #
        # # Get total sequence length
        # T_total = states_seq.size(1)
        #
        # # --- 1. PRE-COMPUTATION (Full Sequence / No Grad) ---
        # # It is safe to process the whole sequence here because we don't backprop
        # with torch.no_grad():
        #     # Process entire sequences through value network
        #     next_values, _ = self.value_net.forward_all_steps(next_states_seq)
        #     next_values = next_values.squeeze(0)
        #
        #     current_values, _ = self.value_net.forward_all_steps(states_seq)
        #     current_values = current_values.squeeze(0)
        #
        #     td_target = rewards + self.gamma * next_values * (1 - dones)
        #     td_delta = td_target - current_values
        #
        #     advantage = compute_gae(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        #     advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        #
        #     # Get old log probs
        #     mu, std, _ = self.actor(states_seq)
        #     mu = mu.squeeze(0)
        #     std = std.squeeze(0)
        #     action_dist = torch.distributions.Normal(mu, std)
        #     old_log_probs = action_dist.log_prob(actions)
        #
        # # --- 2. PPO UPDATE WITH TBPTT ---
        # for _ in range(self.epochs):
        #
        #     # Initialize hidden states to None (or zeros) at the start of the trajectory
        #     # NOTE: Assuming your actor/critic forward() accepts optional hidden states
        #     actor_hidden = None
        #     critic_hidden = None
        #
        #     # Loop through the sequence in chunks
        #     for t in range(0, T_total, bptt_batch_size):
        #
        #         # A. Get indices for this chunk
        #         end_t = min(t + bptt_batch_size, T_total)
        #
        #         # B. Slice the Tensors for this chunk
        #         # We keep the batch dim (1) for the model input: (1, chunk_len, dim)
        #         states_chunk = states_seq[:, t:end_t, :]
        #
        #         # Flattened targets for loss calculation
        #         actions_chunk = actions[t:end_t]
        #         old_log_probs_chunk = old_log_probs[t:end_t]
        #         advantage_chunk = advantage[t:end_t]
        #         td_target_chunk = td_target[t:end_t]
        #
        #         # C. DETACH HIDDEN STATES
        #         # This is the Truncated BPTT step.
        #         # We keep the *values* of the memory, but cut the *gradient* history.
        #         if actor_hidden is not None:
        #             # LSTM hidden state is a tuple (h, c)
        #             actor_hidden = (actor_hidden[0].detach(), actor_hidden[1].detach())
        #
        #         if critic_hidden is not None:
        #             critic_hidden = (critic_hidden[0].detach(), critic_hidden[1].detach())
        #
        #         # D. Forward Pass (Actor)
        #         # Ensure your actor returns the new hidden state
        #         mu, std, actor_hidden = self.actor(states_chunk, actor_hidden)
        #         mu = mu.squeeze(0)
        #         std = std.squeeze(0)
        #
        #         action_dist = torch.distributions.Normal(mu, std)
        #         log_probs = action_dist.log_prob(actions_chunk)
        #
        #         # E. PPO Loss Calculation (Same as before)
        #         log_ratio = (log_probs - old_log_probs_chunk).clamp(-20, 20)
        #         ratio = torch.exp(log_ratio)
        #
        #         surr1 = ratio * advantage_chunk
        #         surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage_chunk
        #         actor_loss = torch.mean(-torch.min(surr1, surr2))
        #
        #         # F. Forward Pass (Critic)
        #         # Ensure your value_net returns the new hidden state
        #         current_values, critic_hidden = self.value_net.forward_all_steps(states_chunk, critic_hidden)
        #         current_values = current_values.squeeze(0)
        #
        #         critic_loss = torch.mean(F.mse_loss(current_values, td_target_chunk))
        #
        #         # G. Update
        #         self.actor_optimizer.zero_grad()
        #         self.critic_optimizer.zero_grad()
        #
        #         actor_loss.backward()
        #         critic_loss.backward()
        #
        #         if not hasattr(self, "critic_loss_history"):
        #             self.critic_loss_history = []
        #         self.critic_loss_history.append(critic_loss.item())
        #
        #         torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        #         torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
        #
        #         self.actor_optimizer.step()
        #         self.critic_optimizer.step()
        #     KL early stopping
            with torch.no_grad():
                approx_kl = (log_probs - old_log_probs).mean()
                if approx_kl > 1.5 * self.kl_tolerance:
                    break
        
        # Decay entropy coefficient after update
        self._decay_entropy_coef()
    
    def _decay_entropy_coef(self):
        """Apply exponential decay to entropy coefficient."""
        self.update_count += 1
        self.entropy_coef = max(
            self.entropy_coef_min,
            self.entropy_coef_initial * (self.entropy_coef_decay ** self.update_count)
        )

    def get_config(self) -> dict:
        """Get agent configuration for saving/loading."""
        config = {
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'act_low': self.act_low.tolist(),
            'act_high': self.act_high.tolist(),
            'gamma': self.gamma,
            'lmbda': self.lmbda,
            'epochs': self.epochs,
            'clip_eps': self.clip_eps,
            'entropy_coef': self.entropy_coef_initial,  # Save initial value
            'entropy_coef_decay': self.entropy_coef_decay,
            'entropy_coef_min': self.entropy_coef_min,
            'use_delta_actions': self.use_delta_actions,
            'max_delta': self.max_delta,
            'use_stacked_obs': self.use_stacked_obs,
        }
        
        if self.use_stacked_obs:
            config.update({
                'stack_size': self.stack_size,
                'hidden_size': self.hidden_size,
                'kernel_size': self.kernel_size,
            })
        else:
            config.update({
                'lstm_hidden_size': self.lstm_hidden_size,
                'num_lstm_layers': self.num_lstm_layers,
            })
        
        return config

    def save(self, path: str):
        """Save agent model parameters and training state."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.value_net.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.get_config(),
            'update_count': self.update_count,
            'current_entropy_coef': self.entropy_coef,
        }, path)

    def load(self, path: str):
        """Load agent model parameters and training state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.value_net.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        # Restore entropy decay state if available (for continuing training)
        if 'update_count' in checkpoint:
            self.update_count = checkpoint['update_count']
        if 'current_entropy_coef' in checkpoint:
            self.entropy_coef = checkpoint['current_entropy_coef']