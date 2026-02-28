# -*- coding: utf-8 -*-
# @Time    : 12/02/2026 16:00
# @Author  : mmai
# @FileName: PPO_hrl.py
# @Software: PyCharm

"""
Hierarchical PPO with Duration-Augmented Actions (Temporal Abstraction).

Architecture:
  Shared backbone (LSTM + Attention) → Two output heads:
    1. Action Head (continuous): delta_width for each gate
    2. Duration Head (discrete): number of env steps to hold the action (1..max_duration)

Why this helps delayed rewards:
  - When the agent commits to an action for k steps, only one "decision point" covers
    those k env steps. The effective MDP horizon shrinks from T to ~T/k.
  - The accumulated reward over k steps is a stronger, less noisy signal.
  - The discount for the next value becomes γ^k, which sharpens the gradient.

Training:
  - The agent stores one "macro-transition" per decision:
      (s_t, delta, k, cumulative_reward_over_k, s_{t+k}, done_{t+k})
  - PPO update treats each macro-transition as a single step.
  - Both heads are updated with the same advantage (cooperative).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from rl.rl_utils import compute_gae, validate_and_save_best, layer_init


def compute_gae_variable_gamma(gamma_base: float, lmbda: float,
                               td_delta: torch.Tensor,
                               durations: torch.Tensor) -> torch.Tensor:
    """GAE with per-step variable discount for macro-transitions.

    In the HRL setting, consecutive macro-transitions span different numbers
    of env steps.  The effective discount between decision i and i+1 is
    γ^{k_i} (not γ), so the GAE recursion becomes:

        A_i = δ_i + (γ^{k_i} · λ) · A_{i+1}

    Args:
        gamma_base: Base discount factor (e.g. 0.99).
        lmbda:      GAE lambda.
        td_delta:   TD errors of shape (T, 1).
        durations:  Duration k_i per macro-step of shape (T, 1).

    Returns:
        advantages: Tensor of shape (T, 1).
    """
    td_delta_np = td_delta.numpy().squeeze(-1)     # (T,)
    durations_np = durations.numpy().squeeze(-1)    # (T,)
    T = len(td_delta_np)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for i in reversed(range(T)):
        # Discount for the transition FROM step i TO step i+1
        gamma_k = gamma_base ** durations_np[i]
        gae = td_delta_np[i] + gamma_k * lmbda * gae
        advantages[i] = gae
    return torch.tensor(advantages, dtype=torch.float).unsqueeze(-1)  # (T, 1)


import math

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# Duration-Augmented Attention Networks
# =============================================================================

class DurationAttentionPolicy(nn.Module):
    """
    Attention-based policy with two heads sharing a backbone:
      1. Action head (continuous): outputs (mean, std) for delta widths
      2. Duration head (discrete): outputs logits over {1, ..., max_duration}

    The backbone is identical to AttentionPolicy (LSTM + MHA).
    """

    def __init__(self, obs_dim, act_dim, hidden_size=64, num_layers=1, num_heads=2,
                 min_std=1e-3, max_std=2.0, max_duration=5):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_links = act_dim // 2  # Each link has 2 actions: front + back gate
        self.features_per_link = obs_dim // self.num_links
        self.hidden_size = hidden_size
        self.min_std = min_std
        self.max_std = max_std
        self.max_duration = max_duration

        # ---- Shared backbone (same as AttentionPolicy) ----
        self.lstm = nn.LSTM(
            input_size=self.features_per_link,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.link_model = nn.Linear(hidden_size, hidden_size)
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

        # ---- Action head (per-link, continuous) ----
        # Each link outputs 2 values: [front_gate_delta, back_gate_delta]
        # self.mean_head = nn.Linear(hidden_size, 2)
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2)
        )
        self.std_head = nn.Linear(hidden_size, 2)

        # ---- Duration head (global, discrete) ----
        # Aggregate link features → global feature → duration logits
        self.duration_fc = nn.Sequential(
            nn.Linear(hidden_size + act_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, max_duration)
        )

    def forward(self, x, hidden=None):
        """
        Args:
            x: (1, seq_len, obs_dim) or (seq_len, obs_dim)
            hidden: LSTM hidden state tuple or None

        Returns:
            mean:       (seq_len, act_dim)  — action means
            std:        (seq_len, act_dim)  — action stds
            dur_logits: (seq_len, max_duration) — duration class logits
            hidden:     updated LSTM hidden state
        """
        if x.dim() == 3:
            x = x.squeeze(0)
        seq_len = x.shape[0]

        # 1. Prepare per-link input (num_links tokens, each with features_per_link)
        x_links = x.view(seq_len, self.num_links, self.features_per_link).transpose(0, 1)

        # 2. Shared LSTM
        lstm_out, hidden_out = self.lstm(x_links, hidden)
        lstm_features = lstm_out.transpose(0, 1)  # (seq_len, num_links, hidden_size)

        # 3. Link projection
        link_features = self.link_model(lstm_features)

        # 4. Inter-link attention
        attn_out, _ = self.attention_layer(
            query=link_features, key=link_features, value=link_features
        )
        coordinated = self.layer_norm(link_features + attn_out)

        # 5a. Action head: each link outputs 2 values [front_gate, back_gate]
        mean = self.mean_head(F.relu(coordinated)).view(seq_len, -1)  # (seq_len, act_dim)
        std = F.softplus(self.std_head(F.relu(coordinated))).view(seq_len, -1).clamp(self.min_std, self.max_std)
        # action_input = torch.cat([F.relu(coordinated), gate_width], dim=-1)  # (seq_len, act_dim, hidden_size+1)
        # mean = self.mean_head(action_input).squeeze(-1)  # (seq_len, act_dim)
        # std = F.softplus(self.std_head(action_input)).squeeze(-1).clamp(self.min_std, self.max_std)

        # 5b. Duration head (global): mean-pool over links → logits
        global_feat = coordinated.mean(dim=1)  # (seq_len, hidden_size)
        dur_logits = self.duration_fc(torch.cat([global_feat, mean], dim=-1))  # (seq_len, max_duration)

        return mean, std, dur_logits, hidden_out


class DurationAttentionValueNetwork(nn.Module):
    """Attention-based value network for MAPPO with duration-augmented actions.

    Used in two modes:
    - **Shared/centralized critic** (standard MAPPO): Takes the concatenated global state
      of all agents as input. Tokens = num_agents * num_links_per_agent.
    - **Per-agent/local critic**: Takes a single agent's local observation as input.
      Tokens = num_links_per_agent (num_agents=1).

    Architecture:
        1. Reshape input → (num_tokens, feat_per_link)
        2. LSTM over time for each token (captures temporal dynamics)
        3. Self-attention over tokens (captures spatial / inter-link interactions)
        4. Global pooling → scalar value

    Fusion options for global pooling:
        - 'attention': Learned query attends to all tokens (default, recommended)
        - 'mean': Simple mean pooling
        - 'max': Max pooling
        - 'mean_max': Concatenate mean and max pooling
        - 'gated': Gated attention pooling with sigmoid weights
    """

    def __init__(self, global_obs_dim, num_agents, num_links_per_agent, feat_per_link,
                 hidden_size=64, num_layers=1, num_heads=2, fusion='attention'):
        """
        Args:
            global_obs_dim: Total dimension of concatenated global state (for validation).
            num_agents: Number of agents in the environment.
            num_links_per_agent: Number of controlled links per agent.
            feat_per_link: Number of features per link (from local observation).
            hidden_size: LSTM and attention hidden dimension.
            num_layers: Number of LSTM layers.
            num_heads: Number of attention heads.
            fusion: Pooling method for global aggregation.
        """
        super().__init__()
        self.global_obs_dim = global_obs_dim
        self.num_agents = num_agents
        self.num_links_per_agent = num_links_per_agent
        self.feat_per_link = feat_per_link
        self.num_tokens = num_agents * num_links_per_agent
        self.hidden_size = hidden_size
        self.fusion = fusion

        # Validate dimensions
        expected_dim = num_agents * num_links_per_agent * feat_per_link
        assert global_obs_dim == expected_dim, (
            f"global_obs_dim ({global_obs_dim}) != num_agents ({num_agents}) * "
            f"num_links_per_agent ({num_links_per_agent}) * feat_per_link ({feat_per_link}) = {expected_dim}"
        )

        # Shared LSTM: processes each (agent, link) token over time
        self.lstm = nn.LSTM(
            input_size=feat_per_link,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Token feature extractor
        self.link_model = nn.Linear(hidden_size, hidden_size)

        # Attention for inter-token (cross-agent, cross-link) coordination
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Fusion-specific layers
        if fusion == 'attention':
            # Learned global query for attention pooling
            self.global_query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
            self.pool_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                batch_first=True
            )
            value_input_dim = hidden_size
        elif fusion == 'mean_max':
            # Concatenate mean and max → 2x hidden_size
            value_input_dim = hidden_size * 2
        elif fusion == 'gated':
            # Gated attention: learn importance weights per token
            self.gate_fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1)
            )
            value_input_dim = hidden_size
        else:  # 'mean' or 'max'
            value_input_dim = hidden_size

        # Global value head
        self.value_head = nn.Linear(hidden_size, 1)
        # self.value_head = nn.Sequential(
        #     nn.Linear(value_input_dim, hidden_size // 2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size // 2, 1)
        # )

    def forward(self, x, hidden=None):
        """
        Args:
            x: (1, seq_len, global_obs_dim) or (seq_len, global_obs_dim)
            hidden: LSTM hidden state tuple or None

        Returns:
            value:  (seq_len, 1)
            hidden: updated (h, c)
        """
        if x.dim() == 3:
            x = x.squeeze(0)  # Only remove batch dim, safe when seq_len == 1
        seq_len = x.shape[0]

        # Reshape global state to (seq_len, num_tokens, feat_per_link)
        # where num_tokens = num_agents * num_links_per_agent
        # Each token represents one (agent, link) pair
        x_tokens = x.view(seq_len, self.num_tokens, self.feat_per_link)

        # Per-token LSTM: transpose to (num_tokens, seq_len, feat_per_link) for LSTM
        x_lstm_input = x_tokens.transpose(0, 1)
        lstm_out, hidden_out = self.lstm(x_lstm_input, hidden)
        lstm_features = lstm_out.transpose(0, 1)  # (seq_len, num_tokens, hidden_size)

        # Token projection
        token_features = self.link_model(lstm_features)

        # Inter-token attention (cross-agent, cross-link interactions)
        attn_out, _ = self.attention(
            query=token_features, key=token_features, value=token_features
        )
        coordinated = self.layer_norm(token_features + attn_out)
        # coordinated: (seq_len, num_tokens, hidden_size)

        # Global aggregation based on fusion type
        if self.fusion == 'attention':
            # Learned query attends to all tokens
            query = self.global_query.expand(seq_len, -1, -1)
            global_state, _ = self.pool_attention(
                query=query, key=coordinated, value=coordinated
            )  # (seq_len, 1, hidden_size)
            global_state = global_state.squeeze(1)  # (seq_len, hidden_size)
        elif self.fusion == 'max':
            global_state = coordinated.max(dim=1)[0]  # (seq_len, hidden_size)
        elif self.fusion == 'mean_max':
            mean_pool = coordinated.mean(dim=1)  # (seq_len, hidden_size)
            max_pool = coordinated.max(dim=1)[0]  # (seq_len, hidden_size)
            global_state = torch.cat([mean_pool, max_pool], dim=-1)  # (seq_len, 2*hidden_size)
        elif self.fusion == 'gated':
            # Compute importance scores for each token
            gate_scores = self.gate_fc(coordinated)  # (seq_len, num_tokens, 1)
            gate_weights = F.softmax(gate_scores, dim=1)  # Softmax over tokens
            global_state = (coordinated * gate_weights).sum(dim=1)  # (seq_len, hidden_size)
        else:  # 'mean'
            global_state = coordinated.mean(dim=1)  # (seq_len, hidden_size)

        # Value estimation
        value = self.value_head(F.elu(global_state))  # (seq_len, 1)

        return value, hidden_out


# =============================================================================
# MAPPO HRL Agent
# =============================================================================

class MAPPOAgentHRL:
    """Multi-Agent PPO agent with temporal abstraction (duration-augmented actions)
    and a centralized critic.

    At each decision point each agent outputs:
      - delta_width  (continuous, one per link/gate)
      - duration k   (discrete, 1..max_duration): how many env steps to hold the action

    The macro-transition stores both local state (for the actor) and the global state 
    (for the centralized critic).
    """

    def __init__(self, obs_dim, global_obs_dim, act_dim, act_low, act_high,
                 num_agents, num_links_per_agent,
                 actor_lr=3e-4, critic_lr=6e-4,
                 gamma=0.99, lmbda=0.95, epochs=10, device="cpu",
                 clip_eps=0.2, entropy_coef=0.01, entropy_coef_decay=0.995,
                 entropy_coef_min=0.001, kl_tolerance=0.01,
                 use_delta_actions=False, max_delta=2.5,
                 lstm_hidden_size=64, num_lstm_layers=1, num_heads=2,
                 use_param_noise=False, param_noise_std=0.1,
                 param_noise_std_min=0.01,
                 use_action_noise=False, action_noise_std=0.01,
                 action_noise_std_min=0.001, num_episodes=100, tm_window=50,
                 use_lr_decay=False, lr_warmup_frac=0.05, lr_min_ratio=0.01,
                 max_duration=5,
                 duration_entropy_coef=0.05,
                 duration_entropy_coef_min=0.001,
                 value_fusion='gated',
                 shared_critic=None,
                 shared_critic_optimizer=None):
        """
        MAPPO agent with temporal abstraction (duration-augmented actions) and centralized critic.

        Args:
            obs_dim: Local observation dimension for this agent.
            global_obs_dim: Dimension of the concatenated global state for the centralized critic.
            act_dim: Action dimension for this agent.
            act_low, act_high: Action bounds.
            num_agents: Total number of agents (for centralized critic tokenization).
            num_links_per_agent: Number of controlled links per agent (assumes uniform).
            shared_critic: Optional shared DurationAttentionValueNetwork instance.
                If provided, all agents share this critic (standard MAPPO).
                If None, each agent creates its own critic (legacy behavior).
            shared_critic_optimizer: Optional shared optimizer for the critic.
                Must be provided if shared_critic is provided.
            ...
        Additional HRL args:
            max_duration: Maximum number of env steps the agent can commit to (default 5).
            duration_entropy_coef: Entropy bonus coefficient for the duration head.
            value_fusion: Pooling method for value network global aggregation.
                Options: 'attention' (default), 'mean', 'max', 'mean_max', 'gated'
        """
        self.obs_dim = obs_dim
        self.global_obs_dim = global_obs_dim
        self.act_dim = act_dim
        self.num_agents = num_agents
        self.num_links_per_agent = num_links_per_agent
        self.act_low = torch.tensor(act_low, dtype=torch.float32)
        self.act_high = torch.tensor(act_high, dtype=torch.float32)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.clip_eps = clip_eps
        self.tm_window = tm_window
        self.tm_step = 1
        self.max_duration = max_duration
        self.duration_entropy_coef = duration_entropy_coef
        self.duration_entropy_coef_initial = duration_entropy_coef
        self.duration_entropy_coef_min = duration_entropy_coef_min
        self.value_fusion = value_fusion

        # Entropy coefficient with exponential decay
        self.entropy_coef_initial = entropy_coef
        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay
        self.entropy_coef_min = entropy_coef_min
        self.update_count = 0

        self.kl_tolerance = kl_tolerance
        self.transition_dict = {
            'states': [], 'global_states': [], 'actions': [],
            'next_states': [], 'next_global_states': [],
            'rewards': [], 'dones': [],
            'durations': [],        # chosen duration k per decision
            'true_rewards': [],
        }

        # Delta action settings
        self.use_delta_actions = use_delta_actions
        self.max_delta = max_delta

        # LSTM settings
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.actor_hidden = None
        self.critic_hidden = None
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        # Derive feat_per_link from local obs structure
        # Local obs is structured as (num_links_per_agent * feat_per_link)
        feat_per_link = obs_dim // num_links_per_agent
        self.feat_per_link = feat_per_link

        # Create actor network (always per-agent / decentralized)
        self.actor = DurationAttentionPolicy(
            obs_dim, act_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            max_std=1,
            max_duration=max_duration,
            num_heads=num_heads
        )

        # Critic network: shared or per-agent
        self._owns_critic = (shared_critic is None)
        if shared_critic is not None:
            # Use shared critic (standard MAPPO) — takes global state as input
            self.value_net = shared_critic
            self.critic_optimizer = shared_critic_optimizer
        else:
            # Per-agent critic — uses LOCAL observation only (not global state)
            self.value_net = DurationAttentionValueNetwork(
                global_obs_dim=obs_dim,          # local obs dimension
                num_agents=1,                     # single agent's links only
                num_links_per_agent=num_links_per_agent,
                feat_per_link=feat_per_link,
                hidden_size=lstm_hidden_size,
                num_layers=num_lstm_layers,
                num_heads=num_heads,
                fusion=value_fusion
            )
            self.critic_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=critic_lr)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.device = device

        # Param noise (adaptive, targets mean_head only)
        self.use_param_noise = use_param_noise
        self.param_noise_std_initial = param_noise_std
        self.param_noise_std = param_noise_std
        self.param_noise_std_min = param_noise_std_min
        self.param_noise_adapt_coef = 1.01  # multiplicative factor for adaptive scaling
        self._param_noise_applied = False
        self._original_mean_head_params = None

        # Action noise
        self.use_action_noise = use_action_noise
        self.action_noise_std_initial = action_noise_std
        self.action_noise_std = action_noise_std
        self.action_noise_std_min = action_noise_std_min
        self.total_updates = num_episodes * 1

        # LR scheduler (only for actor if critic is shared)
        self.use_lr_decay = use_lr_decay
        self.lr_warmup_frac = lr_warmup_frac
        self.lr_min_ratio = lr_min_ratio
        if self.use_lr_decay:
            self._setup_lr_scheduler()

        # Duration commitment tracking (used by validation/evaluation)
        # Initialize here to prevent AttributeError if take_action called before reset_buffer
        self._remaining_duration = 0
        self._committed_action = None

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------
    def reset_buffer(self):
        """Clear rollout buffer, reset hidden states, apply param noise."""
        self.transition_dict = {
            'states': [], 'global_states': [], 'actions': [],
            'next_states': [], 'next_global_states': [],
            'rewards': [], 'dones': [],
            'durations': [],
            'true_rewards': [],
        }
        self.actor_hidden = None
        self.critic_hidden = None
        self.tm_step = 1
        # Duration commitment tracking (used by validation/evaluation)
        self._remaining_duration = 0
        self._committed_action = None
        if self.use_param_noise:
            self._apply_param_noise()

    def init_batch_buffer(self):
        self.batch_buffer = []

    def store_trajectory(self):
        """Store current trajectory to batch buffer."""
        if not hasattr(self, 'batch_buffer'):
            self.init_batch_buffer()
        if len(self.transition_dict['states']) > 0:
            trajectory = {
                'states': np.array(self.transition_dict['states']),
                'global_states': np.array(self.transition_dict['global_states']),
                'actions': np.array(self.transition_dict['actions']),
                'next_states': np.array(self.transition_dict['next_states']),
                'next_global_states': np.array(self.transition_dict['next_global_states']),
                'rewards': np.array(self.transition_dict['rewards']),
                'dones': np.array(self.transition_dict['dones']),
                'durations': np.array(self.transition_dict['durations']),
            }
            self.batch_buffer.append(trajectory)

    def clear_batch_buffer(self):
        self.batch_buffer = []

    def get_batch_size(self):
        if not hasattr(self, 'batch_buffer'):
            return 0
        return len(self.batch_buffer)

    def store_transition(self, state, global_state, action, next_state, next_global_state, reward, done,
                         duration=1, true_reward=None):
        """Store a macro-transition.

        Args:
            state: local observation at decision point
            global_state: combined observations at decision point for centralized critic
            action: continuous action (delta widths)
            next_state: local observation after k env steps
            next_global_state: global observation after k env steps
            reward: *cumulative* global reward over k steps
            done: whether episode ended during the k steps
            duration: chosen k
            true_reward: unnormalized cumulative reward (for logging)
        """
        self.transition_dict['states'].append(state)
        self.transition_dict['global_states'].append(global_state)
        self.transition_dict['actions'].append(action)
        self.transition_dict['next_states'].append(next_state)
        self.transition_dict['next_global_states'].append(next_global_state)
        self.transition_dict['rewards'].append(reward)
        self.transition_dict['dones'].append(done)
        self.transition_dict['durations'].append(duration)
        self.transition_dict['true_rewards'].append(
            true_reward if true_reward is not None else reward)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def take_action(self, state, deterministic=False, return_distribution=False,
                    return_duration=False):
        """
        Select continuous action AND discrete duration.

        When called from validation/evaluation (default mode, no return_duration
        or return_distribution), the agent internally tracks duration commitment:
        it picks an action + duration, then replays the same action for the next
        (duration - 1) calls.  This ensures validation mirrors the training
        behaviour where the action is held for k env steps.

        When called from the HRL training loop with return_distribution=True,
        the commitment tracking is bypassed (the training loop manages duration
        stepping itself).

        Returns:
            Default:               action  (np.ndarray), with internal duration tracking
            return_duration=True:  (action, duration)
            return_distribution=True: (action, duration, mu, sigma, dur_probs)
        """
        # --- Duration commitment replay (validation / evaluation mode) ----
        # If we are still within a committed action period, replay immediately
        # without running the network.  Only applies to default call mode.
        if (not return_distribution and not return_duration
                and self._remaining_duration > 0):
            self._remaining_duration -= 1
            return self._committed_action

        # --- Run policy network ---
        state_array = np.array(state)
        state_tensor = torch.tensor(state_array, dtype=torch.float).unsqueeze(0).to(self.device)

        mu, sigma, dur_logits, self.actor_hidden = self.actor(state_tensor, self.actor_hidden)
        self.tm_step += 1

        # --- Continuous action ---
        if deterministic:
            action = mu
        else:
            action_dist = torch.distributions.Normal(mu, sigma)
            action = action_dist.sample()
            if self.use_action_noise:
                action = action + torch.randn_like(action) * self.action_noise_std

        if self.use_delta_actions:
            delta = torch.clamp(action, -self.max_delta, self.max_delta)
            action_out = delta.cpu().detach().numpy().squeeze()
        else:
            action = torch.clamp(action, self.act_low, self.act_high)
            action_out = action.cpu().detach().numpy().squeeze()

        # --- Discrete duration ---
        dur_probs = F.softmax(dur_logits, dim=-1).squeeze(0)  # (max_duration,)
        if deterministic:
            # Original top-k sampling for duration (kept for reference):
            # duration = int(dur_probs.argmax().item()) + 1  # 1-indexed
            # sample top k
            top_k = 2
            top_k_probs, top_k_indices = dur_probs.topk(top_k)
            # Normalize top_k probabilities to sum to 1
            top_k_probs_normalized = top_k_probs / top_k_probs.sum()
            duration_idx = np.random.choice(
                top_k_indices.detach().cpu().numpy(),
                size=1,
                p=top_k_probs_normalized.detach().cpu().numpy(),
            )[0]
            duration = int(duration_idx) + 1
            # print(duration)

            # New: temperature-scaled sampling over full duration distribution for
            # mostly-stable but slightly stochastic durations at inference.
            # tau = max(getattr(self, "duration_temperature_eval", 0.7), 1e-3)
            # dur_probs_eval = F.softmax(dur_logits / tau, dim=-1).detach().squeeze(0)
            # dur_dist = torch.distributions.Categorical(dur_probs_eval)
            # duration = int(dur_dist.sample().item()) + 1  # 1-indexed
        else:
            dur_dist = torch.distributions.Categorical(dur_probs)
            duration = int(dur_dist.sample().item()) + 1  # 1-indexed

        # --- Return based on call mode ---
        if return_distribution:
            # Training call — caller manages duration stepping
            mu_out = mu.cpu().detach().numpy().squeeze()
            sigma_out = sigma.cpu().detach().numpy().squeeze()
            dur_probs_out = dur_probs.cpu().detach().numpy()
            return action_out, duration, mu_out, sigma_out, dur_probs_out

        if return_duration:
            return action_out, duration

        # Default (validation/evaluation) — commit to action for k steps
        self._committed_action = action_out
        self._remaining_duration = duration - 1  # -1 because this call counts as step 1
        return action_out

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------
    def update_actor_only(self, trajectory_data, total_timesteps):
        """Update only the actor network (used when critic is shared).
        
        Args:
            trajectory_data: List of preprocessed trajectory dicts with advantages/targets.
            total_timesteps: Total timesteps across all trajectories for weighting.
        """
        for epoch in range(self.epochs):
            self.actor_optimizer.zero_grad()
            epoch_kl_exceeded = False

            for data in trajectory_data:
                states_seq = data['states_seq']
                actions = data['actions']
                advantage = data['advantage']
                old_action_lp = data['old_action_log_probs']
                old_dur_lp = data['old_dur_log_probs']
                dur_indices = data['dur_indices']
                T_traj = data['T']
                traj_weight = float(T_traj) / float(total_timesteps)

                actor_hidden = None

                for t in range(0, T_traj, self.tm_window):
                    end_t = min(t + self.tm_window, T_traj)
                    chunk_len = end_t - t
                    chunk_weight = float(chunk_len) / float(T_traj) * traj_weight

                    # Slices
                    s_chunk = states_seq[:, t:end_t, :]
                    a_chunk = actions[t:end_t]
                    old_alp_chunk = old_action_lp[t:end_t]
                    old_dlp_chunk = old_dur_lp[t:end_t]
                    adv_chunk = advantage[t:end_t]
                    dur_chunk = dur_indices[t:end_t]

                    # Detach hidden
                    if actor_hidden is not None:
                        actor_hidden = (actor_hidden[0].detach(), actor_hidden[1].detach())

                    # ---- Actor forward ----
                    mu, std, dur_logits, actor_hidden = self.actor(s_chunk, actor_hidden)
                    mu = mu.squeeze(0)
                    std = std.squeeze(0)

                    # Action distribution
                    act_dist = torch.distributions.Normal(mu, std)
                    act_entropy = act_dist.entropy().mean()
                    act_lp = act_dist.log_prob(a_chunk)

                    # Duration distribution
                    dur_probs = F.softmax(dur_logits.squeeze(0), dim=-1)
                    dur_dist = torch.distributions.Categorical(dur_probs)
                    dur_entropy = dur_dist.entropy().mean()
                    dur_lp = dur_dist.log_prob(dur_chunk).unsqueeze(-1)

                    # Action head ratio & clipped surrogate
                    act_log_ratio = (act_lp - old_alp_chunk).clamp(-20, 20)
                    act_log_ratio_mean = act_log_ratio.mean(dim=-1, keepdim=True)
                    act_ratio = torch.exp(act_log_ratio_mean)
                    act_surr1 = act_ratio * adv_chunk
                    act_surr2 = torch.clamp(act_ratio, 1 - self.clip_eps,
                                            1 + self.clip_eps) * adv_chunk
                    act_loss = torch.mean(-torch.min(act_surr1, act_surr2))

                    # Duration head ratio & clipped surrogate
                    dur_log_ratio = (dur_lp - old_dlp_chunk).clamp(-20, 20)
                    dur_ratio = torch.exp(dur_log_ratio)
                    dur_surr1 = dur_ratio * adv_chunk
                    dur_surr2 = torch.clamp(dur_ratio, 1 - self.clip_eps,
                                            1 + self.clip_eps) * adv_chunk
                    dur_loss = torch.mean(-torch.min(dur_surr1, dur_surr2))

                    # Combined actor loss
                    actor_loss = (
                        act_loss + dur_loss
                        - self.entropy_coef * act_entropy
                        - self.duration_entropy_coef * dur_entropy
                    ) * chunk_weight

                    actor_loss.backward()

                    # KL early stopping
                    with torch.no_grad():
                        approx_kl = act_log_ratio.mean()
                        if approx_kl > 1.5 * self.kl_tolerance:
                            epoch_kl_exceeded = True
                            break

                if epoch_kl_exceeded:
                    break

            if epoch_kl_exceeded:
                break

            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            self.actor_optimizer.step()

    def precompute_trajectory_data(self):
        """Precompute TD targets and advantages for all trajectories in batch buffer.
        
        Returns:
            trajectory_data: List of dicts with preprocessed data for each trajectory.
            all_advantages: List of advantage tensors (for global normalization).
        """
        if not hasattr(self, 'batch_buffer') or len(self.batch_buffer) == 0:
            return [], []

        trajectory_data = []
        all_advantages = []

        with torch.no_grad():
            for traj in self.batch_buffer:
                states = torch.tensor(traj['states'], dtype=torch.float).to(self.device)
                global_states = torch.tensor(traj['global_states'], dtype=torch.float).to(self.device)
                actions = torch.tensor(traj['actions']).view(-1, self.act_dim).to(self.device)
                rewards = torch.tensor(traj['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
                next_global_states = torch.tensor(traj['next_global_states'], dtype=torch.float).to(self.device)
                dones = torch.tensor(traj['dones'], dtype=torch.float).view(-1, 1).to(self.device)
                durations = torch.tensor(traj['durations'], dtype=torch.float).view(-1, 1).to(self.device)

                states_seq = states.unsqueeze(0)
                global_states_seq = global_states.unsqueeze(0)
                next_global_states_seq = next_global_states.unsqueeze(0)
                T_traj = states_seq.size(1)

                # Choose critic input: local obs for per-agent critic, global state for shared critic
                if self._owns_critic:
                    # Per-agent critic uses local observation
                    critic_input_seq = states_seq
                    next_states = torch.tensor(traj['next_states'], dtype=torch.float).to(self.device)
                    next_critic_input_seq = next_states.unsqueeze(0)
                else:
                    # Shared critic uses global state
                    critic_input_seq = global_states_seq
                    next_critic_input_seq = next_global_states_seq

                # Value estimates
                next_values, _ = self.value_net(next_critic_input_seq)
                next_values = next_values.squeeze(0)
                current_values, _ = self.value_net(critic_input_seq)
                current_values = current_values.squeeze(0)

                # Old log probs for actor
                mu, std, dur_logits, _ = self.actor(states_seq)
                mu = mu.squeeze(0)
                std = std.squeeze(0)
                action_dist = torch.distributions.Normal(mu, std)
                old_action_log_probs = action_dist.log_prob(actions)

                dur_probs = F.softmax(dur_logits.squeeze(0), dim=-1)
                dur_indices = (durations - 1).long().squeeze(-1)
                dur_dist = torch.distributions.Categorical(dur_probs)
                old_dur_log_probs = dur_dist.log_prob(dur_indices).unsqueeze(-1)

                # TD target with variable discount
                gamma_k = self.gamma ** durations
                td_target = rewards + gamma_k * next_values * (1 - dones)
                td_delta = td_target - current_values

                advantage = compute_gae_variable_gamma(
                    self.gamma, self.lmbda,
                    td_delta.cpu(), durations.cpu()
                ).to(self.device)

                trajectory_data.append({
                    'states_seq': states_seq,
                    'global_states_seq': global_states_seq,
                    'critic_input_seq': critic_input_seq,
                    'actions': actions,
                    'td_target': td_target,
                    'advantage': advantage,
                    'old_action_log_probs': old_action_log_probs,
                    'old_dur_log_probs': old_dur_log_probs,
                    'dur_indices': dur_indices,
                    'T': T_traj,
                })
                all_advantages.append(advantage)

        return trajectory_data, all_advantages

    def update_batch(self, skip_critic=False):
        """PPO update with TBPTT over macro-transitions.

        Each macro-transition stores (s, a, k, R_cum, s', done).
        The TD target uses γ^k for discounting:
            target_i = R_cum_i + γ^{k_i} * V(s'_i) * (1 - done_i)
            
        Args:
            skip_critic: If True, only update actor (used when critic is shared and
                updated separately). Default False for backward compatibility.
        """
        if not hasattr(self, 'batch_buffer') or len(self.batch_buffer) == 0:
            print("Warning: No trajectories in batch buffer. Skipping update.")
            return

        if self.use_param_noise and self._param_noise_applied:
            self._adapt_param_noise_std()

        # --- Precompute targets (no_grad) ---
        trajectory_data, all_advantages = self.precompute_trajectory_data()
        
        if not trajectory_data:
            return

        # Global advantage normalization
        all_adv = torch.cat(all_advantages, dim=0)
        g_mean, g_std = all_adv.mean(), all_adv.std() + 1e-8
        for d in trajectory_data:
            d['advantage'] = (d['advantage'] - g_mean) / g_std

        # Restore clean actor params now that old log probs are captured
        if self.use_param_noise and self._param_noise_applied:
            self._restore_actor_params()

        total_timesteps = sum(d['T'] for d in trajectory_data)

        if skip_critic:
            # Only update actor (critic is shared and updated separately)
            self.update_actor_only(trajectory_data, total_timesteps)
        else:
            # Update both actor and critic (legacy behavior / per-agent critic)
            self._update_actor_and_critic(trajectory_data, total_timesteps)

        self.clear_batch_buffer()
        self.update_count += 1
        self._decay_entropy_coef()
        self._step_lr_scheduler()
        if self.use_action_noise:
            self._decay_action_noise_std()

    def _update_actor_and_critic(self, trajectory_data, total_timesteps):
        """Update both actor and critic networks (used when critic is per-agent)."""
        for epoch in range(self.epochs):
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            epoch_kl_exceeded = False

            for data in trajectory_data:
                states_seq = data['states_seq']
                global_states_seq = data['global_states_seq']
                actions = data['actions']
                td_target = data['td_target']
                advantage = data['advantage']
                old_action_lp = data['old_action_log_probs']
                old_dur_lp = data['old_dur_log_probs']
                dur_indices = data['dur_indices']
                T_traj = data['T']
                traj_weight = float(T_traj) / float(total_timesteps)

                actor_hidden = None
                critic_hidden = None

                for t in range(0, T_traj, self.tm_window):
                    end_t = min(t + self.tm_window, T_traj)
                    chunk_len = end_t - t
                    chunk_weight = float(chunk_len) / float(T_traj) * traj_weight

                    # Slices
                    s_chunk = states_seq[:, t:end_t, :]
                    critic_chunk = data['critic_input_seq'][:, t:end_t, :]
                    a_chunk = actions[t:end_t]
                    old_alp_chunk = old_action_lp[t:end_t]
                    old_dlp_chunk = old_dur_lp[t:end_t]
                    adv_chunk = advantage[t:end_t]
                    tdt_chunk = td_target[t:end_t]
                    dur_chunk = dur_indices[t:end_t]

                    # Detach hidden
                    if actor_hidden is not None:
                        actor_hidden = (actor_hidden[0].detach(), actor_hidden[1].detach())
                    if critic_hidden is not None:
                        critic_hidden = (critic_hidden[0].detach(), critic_hidden[1].detach())

                    # ---- Actor forward ----
                    mu, std, dur_logits, actor_hidden = self.actor(s_chunk, actor_hidden)
                    mu = mu.squeeze(0)
                    std = std.squeeze(0)

                    # Action distribution
                    act_dist = torch.distributions.Normal(mu, std)
                    act_entropy = act_dist.entropy().mean()
                    act_lp = act_dist.log_prob(a_chunk)

                    # Duration distribution
                    dur_probs = F.softmax(dur_logits.squeeze(0), dim=-1)
                    dur_dist = torch.distributions.Categorical(dur_probs)
                    dur_entropy = dur_dist.entropy().mean()
                    dur_lp = dur_dist.log_prob(dur_chunk).unsqueeze(-1)

                    # Action head ratio & clipped surrogate
                    act_log_ratio = (act_lp - old_alp_chunk).clamp(-20, 20)
                    act_log_ratio_mean = act_log_ratio.mean(dim=-1, keepdim=True)
                    act_ratio = torch.exp(act_log_ratio_mean)
                    act_surr1 = act_ratio * adv_chunk
                    act_surr2 = torch.clamp(act_ratio, 1 - self.clip_eps,
                                            1 + self.clip_eps) * adv_chunk
                    act_loss = torch.mean(-torch.min(act_surr1, act_surr2))

                    # Duration head ratio & clipped surrogate
                    dur_log_ratio = (dur_lp - old_dlp_chunk).clamp(-20, 20)
                    dur_ratio = torch.exp(dur_log_ratio)
                    dur_surr1 = dur_ratio * adv_chunk
                    dur_surr2 = torch.clamp(dur_ratio, 1 - self.clip_eps,
                                            1 + self.clip_eps) * adv_chunk
                    dur_loss = torch.mean(-torch.min(dur_surr1, dur_surr2))

                    # Combined actor loss
                    actor_loss = (
                        act_loss + dur_loss
                        - self.entropy_coef * act_entropy
                        - self.duration_entropy_coef * dur_entropy
                    ) * chunk_weight

                    # ---- Critic forward ----
                    curr_val, critic_hidden = self.value_net(critic_chunk, critic_hidden)
                    critic_loss = torch.mean(F.mse_loss(curr_val, tdt_chunk)) * chunk_weight

                    actor_loss.backward()
                    critic_loss.backward()

                    # KL early stopping
                    with torch.no_grad():
                        approx_kl = act_log_ratio.mean()
                        if approx_kl > 1.5 * self.kl_tolerance:
                            epoch_kl_exceeded = True
                            break

                if epoch_kl_exceeded:
                    break

            if epoch_kl_exceeded:
                break

            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    # ------------------------------------------------------------------
    # Decay helpers (same as PPO_tbptt)
    # ------------------------------------------------------------------
    def _decay_entropy_coef(self):
        if self.entropy_coef_initial > 0 and self.entropy_coef_min > 0:
            decay_rate = (self.entropy_coef_min / self.entropy_coef_initial) ** (
                1.0 / max(self.total_updates, 1))
            self.entropy_coef = max(
                self.entropy_coef_initial * (decay_rate ** self.update_count),
                self.entropy_coef_min)

            # decay duration entropy coef
            decay_rate_duration = (self.duration_entropy_coef_min / self.duration_entropy_coef_initial) ** (
                1.0 / max(self.total_updates, 1))
            self.duration_entropy_coef = max(
                self.duration_entropy_coef_initial * (decay_rate_duration ** self.update_count),
                self.duration_entropy_coef_min)
        else:
            progress = min(self.update_count / self.total_updates, 1.0)
            self.entropy_coef = self.entropy_coef_initial + \
                (self.entropy_coef_min - self.entropy_coef_initial) * progress

    def _setup_lr_scheduler(self):
        warmup_steps = max(1, int(self.total_updates * self.lr_warmup_frac))
        total = max(warmup_steps + 1, int(self.total_updates))
        min_ratio = self.lr_min_ratio

        def lr_lambda(step):
            if step < warmup_steps:
                return min_ratio + (1.0 - min_ratio) * (step / warmup_steps)
            else:
                decay_steps = total - warmup_steps
                progress = min((step - warmup_steps) / decay_steps, 1.0)
                return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))

        self.actor_scheduler = torch.optim.lr_scheduler.LambdaLR(self.actor_optimizer, lr_lambda)
        self.critic_scheduler = torch.optim.lr_scheduler.LambdaLR(self.critic_optimizer, lr_lambda)

    def _step_lr_scheduler(self):
        if not self.use_lr_decay:
            return
        self.actor_scheduler.step()
        self.critic_scheduler.step()

    def get_current_lr(self):
        if self.use_lr_decay:
            return {
                'actor_lr': self.actor_scheduler.get_last_lr()[0],
                'critic_lr': self.critic_scheduler.get_last_lr()[0],
            }
        return {'actor_lr': self.actor_lr, 'critic_lr': self.critic_lr}

    # Param noise helpers — targets mean_head only, with adaptive std scaling
    def _apply_param_noise(self):
        """Add Gaussian noise to mean_head parameters only for exploration."""
        if self._param_noise_applied:
            self._restore_actor_params()
        # Store only mean_head original params
        self._original_mean_head_params = {
            name: param.data.clone()
            for name, param in self.actor.mean_head.named_parameters()
        }
        with torch.no_grad():
            for name, param in self.actor.mean_head.named_parameters():
                if param.requires_grad:
                    param.data.add_(torch.randn_like(param) * self.param_noise_std)
        self._param_noise_applied = True

    def _restore_actor_params(self):
        """Restore original mean_head parameters (remove noise)."""
        if self._original_mean_head_params is not None:
            with torch.no_grad():
                for name, param in self.actor.mean_head.named_parameters():
                    if name in self._original_mean_head_params:
                        param.data.copy_(self._original_mean_head_params[name])
            self._original_mean_head_params = None
        self._param_noise_applied = False

    def _adapt_param_noise_std(self):
        """Adaptively scale param_noise_std based on KL between perturbed and
        unperturbed policies over the collected batch.

        Must be called *before* _restore_actor_params() so that the current
        actor still has noisy mean_head weights.
        """
        if not hasattr(self, 'batch_buffer') or len(self.batch_buffer) == 0:
            return

        with torch.no_grad():
            # --- Get perturbed policy outputs (current noisy mean_head) ---
            all_perturbed_mu = []
            all_perturbed_std = []
            all_states_seq = []
            for traj in self.batch_buffer:
                states = torch.tensor(traj['states'], dtype=torch.float).to(self.device)
                states_seq = states.unsqueeze(0)  # (1, T, obs_dim)
                mu_p, std_p, _, _ = self.actor(states_seq)
                all_perturbed_mu.append(mu_p.squeeze(0))
                all_perturbed_std.append(std_p.squeeze(0))
                all_states_seq.append(states_seq)

            # --- Temporarily restore clean params to get unperturbed outputs ---
            self._restore_actor_params()

            all_clean_mu = []
            all_clean_std = []
            for states_seq in all_states_seq:
                mu_c, std_c, _, _ = self.actor(states_seq)
                all_clean_mu.append(mu_c.squeeze(0))
                all_clean_std.append(std_c.squeeze(0))

            # --- Compute mean KL(clean || perturbed) ---
            # KL between two diagonal Gaussians:
            #   KL(p||q) = log(σ_q/σ_p) + (σ_p² + (μ_p - μ_q)²) / (2σ_q²) - 0.5
            total_kl = 0.0
            total_n = 0
            for mu_c, std_c, mu_p, std_p in zip(
                    all_clean_mu, all_clean_std, all_perturbed_mu, all_perturbed_std):
                kl = (torch.log(std_p / std_c)
                      + (std_c ** 2 + (mu_c - mu_p) ** 2) / (2.0 * std_p ** 2)
                      - 0.5)
                total_kl += kl.sum().item()
                total_n += kl.numel()

            mean_kl = total_kl / max(total_n, 1)

            # --- Adaptive scaling ---
            # Use kl_tolerance as the target KL for noise adaptation
            if mean_kl > self.kl_tolerance:
                # Too much perturbation → shrink noise
                self.param_noise_std = max(
                    self.param_noise_std / self.param_noise_adapt_coef,
                    self.param_noise_std_min)
            else:
                # Room for more exploration → grow noise
                self.param_noise_std *= self.param_noise_adapt_coef

        # Re-apply noise (params were restored above for KL computation)
        # Note: _param_noise_applied was set to False inside _restore_actor_params
        # The caller (update_batch) will call _restore_actor_params again which is
        # now a no-op, and training proceeds with clean params.

    def _decay_action_noise_std(self):
        progress = min(self.update_count / self.total_updates, 1.0)
        self.action_noise_std = self.action_noise_std_initial + \
            (self.action_noise_std_min - self.action_noise_std_initial) * progress

    def reset_optimizer(self):
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.critic_lr)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def get_config(self) -> dict:
        config = {
            'obs_dim': self.obs_dim,
            'global_obs_dim': self.global_obs_dim,
            'act_dim': self.act_dim,
            'act_low': self.act_low.tolist(),
            'act_high': self.act_high.tolist(),
            'num_agents': self.num_agents,
            'num_links_per_agent': self.num_links_per_agent,
            'gamma': self.gamma,
            'lmbda': self.lmbda,
            'epochs': self.epochs,
            'clip_eps': self.clip_eps,
            'entropy_coef': self.entropy_coef_initial,
            'entropy_coef_decay': self.entropy_coef_decay,
            'entropy_coef_min': self.entropy_coef_min,
            'use_delta_actions': self.use_delta_actions,
            'max_delta': self.max_delta,
            'use_param_noise': self.use_param_noise,
            'param_noise_std': self.param_noise_std_initial,
            'param_noise_std_min': self.param_noise_std_min,
            'use_action_noise': self.use_action_noise,
            'action_noise_std': self.action_noise_std_initial,
            'action_noise_std_min': self.action_noise_std_min,
            'total_updates': self.total_updates,
            'use_lr_decay': self.use_lr_decay,
            'lr_warmup_frac': self.lr_warmup_frac,
            'lr_min_ratio': self.lr_min_ratio,
            'lstm_hidden_size': self.lstm_hidden_size,
            'num_lstm_layers': self.num_lstm_layers,
            'max_duration': self.max_duration,
            'duration_entropy_coef': self.duration_entropy_coef,
            'value_fusion': self.value_fusion,
        }
        return config

    def save(self, path: str):
        if self.use_param_noise and self._param_noise_applied:
            self._restore_actor_params()
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.value_net.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.get_config(),
            'update_count': self.update_count,
            'current_entropy_coef': self.entropy_coef,
            'current_param_noise_std': self.param_noise_std if self.use_param_noise else None,
            'current_action_noise_std': self.action_noise_std if self.use_action_noise else None,
        }
        # Save LR scheduler state if using decay
        if self.use_lr_decay:
            save_dict['actor_scheduler_state_dict'] = self.actor_scheduler.state_dict()
            save_dict['critic_scheduler_state_dict'] = self.critic_scheduler.state_dict()
        torch.save(save_dict, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.value_net.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        if 'update_count' in checkpoint:
            self.update_count = checkpoint['update_count']
        if 'current_entropy_coef' in checkpoint:
            self.entropy_coef = checkpoint['current_entropy_coef']
        if 'current_param_noise_std' in checkpoint and checkpoint['current_param_noise_std'] is not None:
            self.param_noise_std = checkpoint['current_param_noise_std']
        if 'current_action_noise_std' in checkpoint and checkpoint['current_action_noise_std'] is not None:
            self.action_noise_std = checkpoint['current_action_noise_std']
        # Load LR scheduler state if using decay and state was saved
        if self.use_lr_decay:
            if 'actor_scheduler_state_dict' in checkpoint:
                self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler_state_dict'])
            if 'critic_scheduler_state_dict' in checkpoint:
                self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler_state_dict'])


# =============================================================================
# Shared Critic Update Function
# =============================================================================

def update_shared_critic(agents: dict, shared_critic, shared_critic_optimizer,
                         epochs: int = 10, tm_window: int = 50, device: str = "cpu"):
    """Update the shared centralized critic using data from all agents.
    
    This function collects trajectory data from all agents and performs a single
    critic update using the combined data. This is more sample-efficient than
    having each agent update its own critic.
    
    Args:
        agents: Dict mapping agent_id -> MAPPOAgentHRL instance.
        shared_critic: The shared DurationAttentionValueNetwork instance.
        shared_critic_optimizer: Optimizer for the shared critic.
        epochs: Number of PPO epochs for critic update.
        tm_window: Truncated BPTT window size.
        device: Device for tensor operations.
    
    Returns:
        critic_loss_avg: Average critic loss over the update.
    """
    # Collect all trajectory data from all agents
    all_trajectory_data = []
    
    for agent_id, agent in agents.items():
        traj_data, _ = agent.precompute_trajectory_data()
        all_trajectory_data.extend(traj_data)
    
    if not all_trajectory_data:
        return 0.0
    
    total_timesteps = sum(d['T'] for d in all_trajectory_data)
    total_critic_loss = 0.0
    num_updates = 0
    
    for epoch in range(epochs):
        shared_critic_optimizer.zero_grad()
        epoch_loss = 0.0
        
        for data in all_trajectory_data:
            global_states_seq = data['global_states_seq']
            td_target = data['td_target']
            T_traj = data['T']
            traj_weight = float(T_traj) / float(total_timesteps)
            
            critic_hidden = None
            
            for t in range(0, T_traj, tm_window):
                end_t = min(t + tm_window, T_traj)
                chunk_len = end_t - t
                chunk_weight = float(chunk_len) / float(T_traj) * traj_weight
                
                gs_chunk = global_states_seq[:, t:end_t, :]
                tdt_chunk = td_target[t:end_t]
                
                if critic_hidden is not None:
                    critic_hidden = (critic_hidden[0].detach(), critic_hidden[1].detach())
                
                curr_val, critic_hidden = shared_critic(gs_chunk, critic_hidden)
                critic_loss = torch.mean(F.mse_loss(curr_val, tdt_chunk)) * chunk_weight
                
                critic_loss.backward()
                epoch_loss += critic_loss.item()
        
        torch.nn.utils.clip_grad_norm_(shared_critic.parameters(), max_norm=0.5)
        shared_critic_optimizer.step()
        
        total_critic_loss += epoch_loss
        num_updates += 1
    
    return total_critic_loss / max(num_updates, 1)
