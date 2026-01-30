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
from rl.rl_utils import compute_gae, save_with_best_return, validate_and_save_best, layer_init
import math
import os
import collections
from .SAC import MLPEncoder, StackedEncoder
from torch_geometric.nn import DenseGATConv
from torch_geometric.utils import to_dense_adj
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

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
    def __init__(self, obs_dim, act_dim, hidden_size=64, num_layers=1):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
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
        # num_gates = self.act_dim
        # gate_hidden_size = 32
        # self.gate_width_head = nn.Linear(num_gates, gate_hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)
        # self.value_head = nn.Linear(hidden_size + gate_hidden_size, 1)

    def forward(self, x, hidden=None):
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
# GAT-LSTM Networks (Temporal -> Spatial)
# =============================================================================

class GATLSTMPolicyNetwork(nn.Module):
    """
    GAT-LSTM policy network: Per-link LSTM -> GAT -> Action heads.

    Architecture:
    1. Each link's temporal features are processed by a shared LSTM
    2. LSTM outputs are aggregated using GAT (spatial attention)
    3. GAT output is used to generate action mean and std

    Args:
        features_per_link: Number of features per link (e.g., 3 for option1)
        act_dim: Action dimension (number of gates to control)
        lstm_hidden_size: Hidden size for LSTM
        gat_hidden_size: Hidden size for GAT layer
        num_heads: Number of attention heads in GAT
        num_lstm_layers: Number of LSTM layers
        min_std, max_std: Bounds for action standard deviation
    """
    def __init__(self, obs_dim, act_dim,
                 lstm_hidden_size=64, gat_hidden_size=64,
                 num_heads=4, num_lstm_layers=1,
                 min_std=1e-3, max_std=10.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.features_per_link = self.obs_dim // act_dim
        self.act_dim = act_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.gat_hidden_size = gat_hidden_size
        self.num_heads = num_heads
        self.num_lstm_layers = num_lstm_layers
        self.min_std = min_std
        self.max_std = max_std

        # Shared LSTM for processing each link's temporal data
        self.lstm = nn.LSTM(
            input_size=self.features_per_link,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True
        )

        # GAT layer for spatial aggregation
        # DenseGATConv is more efficient for small fully-connected graphs
        self.gat = DenseGATConv(
            in_channels=lstm_hidden_size,
            out_channels=gat_hidden_size,
            heads=num_heads,
            dropout=0.0,
            concat=True  # Concatenate heads instead of averaging
        )

        # After GAT, we have (gat_hidden_size * num_heads) features per node
        gat_output_dim = gat_hidden_size * num_heads

        # Action heads
        self.mean_head = nn.Linear(gat_output_dim, act_dim)
        self.std_head = nn.Linear(gat_output_dim, act_dim)

        # Hidden states for LSTM (will be managed externally for each link)
        self.lstm_hidden = None

        # Cache for dense adjacency matrix (created on first forward pass)
        self._cached_adj = None
        self._cached_num_links = None

    def forward(self, x, adj=None, hidden=None):
        """
        Forward pass with graph structure.

        Args:
            x: Observations of shape (seq_len, num_links * features_per_link)
                or (batch, seq_len, num_links * features_per_link) for sequences
                batch_size is always 1
            adj: Dense adjacency matrix (optional). If None, uses fully connected graph.
                 Shape: (batch, num_links, num_links)
            hidden: LSTM hidden states (optional) - dict with keys being link indices

        Returns:
            mean: Action mean (batch, act_dim)
            std: Action std (batch, act_dim)
            hidden: Updated LSTM hidden states
        """
        seq_len = x.shape[0]
        # batch_size = 1

        # Reshape input to (batch, num_links, features_per_link)
        # Assume x is flat: (batch, num_links * features_per_link)
        x_reshaped = x.view(seq_len, self.act_dim, self.features_per_link)

        # Add sequence dimension if not present: (batch, num_links, 1, features)
        # if x_reshaped.dim() == 3:
        #     x_reshaped = x_reshaped.unsqueeze(0)  # (batch, seq_len, num_links, features)

        # Process each link through LSTM independently
        x_lstm_input = x_reshaped.transpose(0, 1) # (num_links, seq_len, features_per_link)

        # LSTM forward (shared weights across all links)
        lstm_out, hidden_out = self.lstm(x_lstm_input, hidden)  # (num_links, lstm_hidden)

        # Take last timestep output
        # lstm_features = lstm_out[:, -1, :]  # (batch*num_links, lstm_hidden)
        lstm_features = lstm_out.transpose(0, 1) # (seq_len, num_links, lstm_hidden_size)

        # Reshape back to (batch, num_links, lstm_hidden)
        # lstm_features = lstm_features.view(batch_size, num_links, self.lstm_hidden_size)

        # Build dense adjacency matrix for DenseGAT
        if adj is None:
            # Use cached fully connected adjacency matrix
            if self._cached_adj is None or self._cached_num_links != self.act_dim:
                # Create fully connected adjacency: all 1s (batch, num_links, num_links)
                self._cached_adj = torch.ones(1, self.act_dim, self.act_dim)
                # 0 for diagonal, 1 for off-diagonal
                self._cached_adj = self._cached_adj * torch.eye(self.act_dim)
                self._cached_num_links = self.act_dim

            adj = self._cached_adj.to(lstm_features.device)

        # Apply DenseGAT: expects (batch, num_nodes, in_channels) and (batch, num_nodes, num_nodes)
        gat_out = self.gat(lstm_features, adj)  # (batch, num_links, gat_hidden*num_heads)

        # gat_out is already (batch, num_links, gat_hidden*num_heads)
        gat_features = gat_out

        # Aggregate across links (e.g., mean pooling) to get global representation
        # Or use specific link features for per-link actions
        # For simplicity, use mean pooling
        global_features = gat_features.mean(dim=1)  # (batch, gat_hidden*num_heads)

        # Generate actions
        mean = self.mean_head(F.relu(global_features))
        std = F.softplus(self.std_head(F.relu(global_features))).clamp(self.min_std, self.max_std)

        return mean, std, hidden_out


class GATLSTMValueNetwork(nn.Module):
    """
    GAT-LSTM value network: Per-link LSTM -> GAT -> Value head.

    Similar to policy network but outputs a single value estimate.
    """
    def __init__(self, obs_dim, num_links,
                 lstm_hidden_size=64, gat_hidden_size=64,
                 num_heads=1, num_lstm_layers=1):
        super().__init__()
        self.obs_dim = obs_dim
        self.features_per_link = self.obs_dim // num_links
        self.num_links = num_links
        self.lstm_hidden_size = lstm_hidden_size
        self.gat_hidden_size = gat_hidden_size
        self.num_heads = num_heads
        self.num_lstm_layers = num_lstm_layers

        # Shared LSTM
        self.lstm = nn.LSTM(
            input_size=self.features_per_link,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True
        )

        # GAT layer (DenseGAT for small fully-connected graphs)
        self.gat = DenseGATConv(
            in_channels=lstm_hidden_size,
            out_channels=gat_hidden_size,
            heads=num_heads,
            dropout=0.0,
            concat=True
        )

        gat_output_dim = gat_hidden_size * num_heads

        # Value head
        self.value_head = nn.Linear(gat_output_dim, 1)

        self.lstm_hidden = None

        # Cache for dense adjacency matrix
        self._cached_adj = None

    def forward(self, x, adj=None, hidden=None):
        """
        Forward pass for value estimation.

        Args:
            x: Observations of shape (seq_len, num_links * features_per_link)
            adj: Dense adjacency matrix (optional)
            hidden: LSTM hidden states (optional)

        Returns:
            value: State value estimate (batch, 1)
            hidden: Updated LSTM hidden states
        """
        # x = x.unsqueeze(0) # (1, seq_len, num_links * features_per_link)
        seq_len = x.shape[0]
        # batch_size = 1

        # Reshape input
        x_lstm_input = x.view(seq_len, self.num_links, self.features_per_link).transpose(0, 1) # (num_links, seq_len, features_per_link)

        # LSTM forward
        lstm_out, hidden_out = self.lstm(x_lstm_input, hidden)
        lstm_features = lstm_out.transpose(0, 1) # (num_links, seq_len, lstm_hidden_size)

        # Reshape to (batch, num_links, lstm_hidden) for DenseGAT
        # lstm_features = lstm_features.unsqueeze(0)  # (1, num_links, lstm_hidden)

        # Build dense adjacency matrix for DenseGAT
        if adj is None:
            if self._cached_adj is None:
                # Create fully connected adjacency: all 1s (batch, num_links, num_links)
                self._cached_adj = torch.ones(1, self.num_links, self.num_links)

            adj = self._cached_adj.to(lstm_features.device)

        # Apply DenseGAT
        gat_out = self.gat(lstm_features, adj)  # (batch, num_links, gat_hidden*num_heads)

        gat_features = gat_out

        # Global pooling
        global_features = gat_features.mean(dim=1)

        # Value estimate
        value = self.value_head(F.relu(global_features))

        return value, hidden_out



# =============================================================================
# Stacked State Networks (similar to SAC)
# =============================================================================

class StackedPolicyNetwork(nn.Module):
    """Policy network for stacked observations."""
    def __init__(self, obs_dim, act_dim, stack_size=4, hidden_size=64, kernel_size=3,
                 min_std=1e-3, max_std=10.0):
        super(StackedPolicyNetwork, self).__init__()
        self.encoder = StackedEncoder(obs_dim, stack_size=stack_size,
                                     hidden_size=hidden_size, kernel_size=kernel_size)
        # self.encoder = MLPEncoder(obs_dim, stack_size=stack_size, hidden_size=hidden_size)
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
        self.encoder = StackedEncoder(obs_dim, stack_size=stack_size,
                                     hidden_size=hidden_size, kernel_size=kernel_size)
        # self.encoder = MLPEncoder(obs_dim, stack_size=stack_size, hidden_size=hidden_size)
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


# =============================================================================
# PPO Agent with upstream and downstream modeling
# =============================================================================
class UDLSTMPolicyNetwork(nn.Module):
    """
    LSTM-based policy network with upstream/downstream link aggregation.
    
    Each link's action is informed by:
    1. Its own temporal features (from shared LSTM)
    2. Aggregated features from all other links (upstream/downstream context)
    """
    def __init__(self, obs_dim, act_dim, hidden_size=64, num_layers=1,
                 min_std=1e-3, max_std=10.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim  # num_links = act_dim
        self.features_per_link = obs_dim // act_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.min_std = min_std
        self.max_std = max_std

        # Shared LSTM for processing each link's temporal data
        self.lstm = nn.LSTM(
            input_size=self.features_per_link,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Link-specific feature extractor
        # self.link_model = nn.Linear(hidden_size, hidden_size)
        self.link_model = nn.Linear(hidden_size, hidden_size)
        
        # Upstream/Downstream aggregation model
        # Input: [link_features, other_links_sum] -> hidden_size
        self.ud_model = nn.Linear(2 * hidden_size, hidden_size)

        # Shared latent layer for action coordination
        self.shared_latent_layer = nn.Linear(hidden_size * act_dim, hidden_size * act_dim)
        # self.shared_latent_layer = nn.Linear((hidden_size+1) * act_dim, hidden_size * act_dim)

        # Per-link action heads (output 1 action per link)
        self.mean_head = nn.Linear(hidden_size, 1)
        self.std_head = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden=None):
        """
        Forward pass with upstream/downstream aggregation.

        Args:
            x: Observations of shape (seq_len, num_links * features_per_link)
            hidden: Optional tuple (h, c) of hidden states

        Returns:
            mean: Action mean (seq_len, act_dim)
            std: Action std (seq_len, act_dim)
            hidden: Updated hidden state tuple (h, c)
        """
        if x.dim() == 3:
            x = x.squeeze()
        seq_len = x.shape[0]
        
        # Reshape input: (seq_len, num_links * features) -> (num_links, seq_len, features)
        x_lstm_input = x.view(seq_len, self.act_dim, self.features_per_link).transpose(0, 1)
        
        # LSTM forward (shared weights across all links)
        lstm_out, hidden_out = self.lstm(x_lstm_input, hidden)  # (num_links, seq_len, hidden_size)
        lstm_features = lstm_out.transpose(0, 1)  # (seq_len, num_links, hidden_size)
        
        # Extract link-specific features
        link_features = self.link_model(lstm_features)  # (seq_len, num_links, hidden_size)
        
        # Compute sum of all link features
        all_links_sum = link_features.sum(dim=1)  # (seq_len, hidden_size)
        
        # For each link, get "other links" features by subtracting its own
        other_links_features = all_links_sum.unsqueeze(1) - link_features  # (seq_len, num_links, hidden_size)
        
        # Concatenate each link's features with aggregated other links' features
        combined_features = torch.cat([link_features, other_links_features], dim=2)  # (seq_len, num_links, 2*hidden_size)
        
        # Process through UD model
        ud_features = self.ud_model(combined_features)  # (seq_len, num_links, hidden_size)
        # get gate width features
        # gate_widths = x.reshape(seq_len, self.act_dim, self.features_per_link)[:,:,-1].unsqueeze(2) # (seq_len, num_links, 1)
        # ud_features = torch.cat([ud_features, gate_widths], dim=2)  # (seq_len, num_links, hidden_size + 1)

        # Flatten features for shared latent layer
        shared_features = ud_features.view(seq_len, -1)  # (seq_len, num_links * hidden_size)
        shared_latent = self.shared_latent_layer(shared_features)  # (seq_len, num_links * hidden_size)
        shared_latent = shared_latent.view(seq_len, self.act_dim, self.hidden_size)  # (seq_len, num_links, hidden_size)

        # Generate per-link actions
        mean = self.mean_head(F.relu(shared_latent)).squeeze(-1)  # (seq_len, num_links)
        std = F.softplus(self.std_head(F.relu(shared_latent))).squeeze(-1).clamp(self.min_std, self.max_std)  # (seq_len, num_links)
        # # Generate per-link actions
        # mean = self.mean_head(F.relu(ud_features)).squeeze(-1)  # (seq_len, num_links)
        # std = F.softplus(self.std_head(F.relu(ud_features))).squeeze(-1).clamp(self.min_std, self.max_std)  # (seq_len, num_links)

        return mean, std, hidden_out


class UDLSTMValueNetwork(nn.Module):
    """Stateful LSTM-based value network that maintains hidden state across timesteps."""
    def __init__(self, obs_dim, act_dim, hidden_size=64, num_layers=1):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_links = act_dim
        self.features_per_link = obs_dim // act_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=self.features_per_link,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.link_model = nn.Linear(hidden_size, hidden_size)
        self.ud_model = nn.Linear(2*hidden_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)
        # self.value_head = nn.Linear(hidden_size + 1, 1)

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
        x = x.squeeze()
        seq_len = x.shape[0]
        # batch_size = 1

        # Reshape input
        x_lstm_input = x.view(seq_len, self.num_links, self.features_per_link).transpose(0, 1) # (num_links, seq_len, features_per_link)

        # LSTM forward
        lstm_out, hidden_out = self.lstm(x_lstm_input, hidden)
        lstm_features = lstm_out.transpose(0, 1)  # (seq_len, num_links, lstm_hidden_size)
        
        # Extract link-specific features
        link_features = self.link_model(lstm_features)  # (seq_len, num_links, hidden_size)
        
        # For each link, aggregate features from all OTHER links
        # Create upstream/downstream features by summing all other links
        
        # Compute sum of all link features: (seq_len, hidden_size)
        all_links_sum = link_features.sum(dim=1)  # (seq_len, hidden_size)
        
        # For each link, subtract its own features to get "other links" sum
        # Broadcast: (seq_len, 1, hidden_size) - (seq_len, num_links, hidden_size)
        other_links_features = all_links_sum.unsqueeze(1) - link_features  # (seq_len, num_links, hidden_size)
        
        # Concatenate each link's features with aggregated other links' features
        combined_features = torch.cat([link_features, other_links_features], dim=2)  # (seq_len, num_links, 2*hidden_size)
        
        # Process through UD model
        ud_features = self.ud_model(combined_features)  # (seq_len, num_links, hidden_size)
        # get gate width features
        # gate_widths = x.reshape(seq_len, self.num_links, self.features_per_link)[:,:,-1].unsqueeze(2) # (seq_len, num_links, 1)
        # ud_features = torch.cat([ud_features, gate_widths], dim=2)  # (seq_len, num_links, hidden_size + 1)

        # Aggregate across links (e.g., mean pooling for global value)
        # global_features = ud_features.mean(dim=1)  # (seq_len, hidden_size)
        global_features = ud_features.mean(dim=1) # (seq_len, hidden_size + 1)
        
        # Compute value
        value = self.value_head(F.elu(global_features))  # (seq_len, 1)

        return value, hidden_out


class AttentionPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64, num_layers=1,
                 min_std=1e-3, max_std=10.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim 
        self.features_per_link = obs_dim // act_dim
        self.hidden_size = hidden_size
        self.min_std = min_std
        self.max_std = max_std

        # Shared LSTM (Kept the same)
        self.lstm = nn.LSTM(
            input_size=self.features_per_link,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Link-specific feature extractor
        self.link_model = nn.Linear(hidden_size, hidden_size)
        
        # --- IMPROVEMENT START ---
        # Instead of a fixed Linear(N*H, N*H), we use Multi-Head Attention.
        # This allows "All-to-All" communication but uses shared weights.
        # It is invariant to the number of links and their order.
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=1,
            batch_first=True
        )
        # --- IMPROVEMENT END ---

        # Layer normalization on coordinated features (per link, per timestep)
        # self.layer_norm = nn.LayerNorm(hidden_size)

        # Per-link action heads (Shared weights, applied per link)
        self.mean_head = nn.Linear(hidden_size, 1)
        self.std_head = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden=None):
        if x.dim() == 3:
            x = x.squeeze()
        seq_len = x.shape[0]
        
        # 1. Prepare Input
        # (seq_len, num_links, features)
        x_lstm_input = x.view(seq_len, self.act_dim, self.features_per_link).transpose(0, 1)
        
        # 2. LSTM (Independent processing)
        lstm_out, hidden_out = self.lstm(x_lstm_input, hidden) 
        lstm_features = lstm_out.transpose(0, 1) # (seq_len, num_links, hidden_size)
        
        # 3. Link Features
        link_features = self.link_model(lstm_features) 

        # 4. Coordination via Attention (The Fix)
        # We treat 'num_links' as the sequence length for the transformer logic.
        # We reshape to mix time/batch so attention happens purely between links at the same timestep.
        
        # Shape: (seq_len * num_links, hidden_size) -> This loses "who is who"
        # We need: (seq_len, num_links, hidden_size)
        
        # Attention expects: (Batch, Sequence, Features)
        # Here "Batch" is actual_seq_len, "Sequence" is num_links
        # This creates "All-to-all" communication between links
        attn_out, _ = self.attention_layer(
            query=link_features, 
            key=link_features, 
            value=link_features
        )
        
        # Residual connection (optional but recommended)
        coordinated_features = link_features + attn_out
        # Apply layer normalization for stability
        # coordinated_features = self.layer_norm(coordinated_features)
        
        # 5. Final Heads
        # The 'coordinated_features' now contains info from self + all other links
        mean = self.mean_head(F.relu(coordinated_features)).squeeze(-1) 
        std = F.softplus(self.std_head(F.relu(coordinated_features))).squeeze(-1).clamp(self.min_std, self.max_std)

        return mean, std, hidden_out

class AttentionValueNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64, num_layers=1):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.features_per_link = obs_dim // act_dim
        self.hidden_size = hidden_size
        
        # 1. Shared LSTM (Same as Policy)
        self.lstm = nn.LSTM(
            input_size=self.features_per_link,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # 2. Link Feature Extractor
        self.link_model = nn.Linear(hidden_size, hidden_size)

        # 3. Attention (The Upgrade)
        # Replaces the manual "sum of others" logic
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=1,
            batch_first=True
        )
        
        # Layer normalization on coordinated features
        # self.layer_norm = nn.LayerNorm(hidden_size)

        # 4. Global Value Head
        # Takes the aggregated system state and outputs 1 value
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden=None):
        """
        Returns:
            value: (seq_len, 1)
            hidden: (h, c)
        """
        if x.dim() == 3:
            x = x.squeeze()
        seq_len = x.shape[0]

        # --- Phase 1: Individual Link Processing ---
        # Reshape: (seq_len, num_links, features)
        x_lstm_input = x.view(seq_len, self.act_dim, self.features_per_link).transpose(0, 1)
        
        # LSTM Forward
        lstm_out, hidden_out = self.lstm(x_lstm_input, hidden)
        lstm_features = lstm_out.transpose(0, 1) # (seq_len, num_links, hidden_size)
        
        # Linear projection
        link_features = self.link_model(lstm_features)

        # --- Phase 2: Coordination (Attention) ---
        # "All-to-All" communication.
        # Links exchange info. If Link A is jammed, Link B knows about it here.
        attn_out, _ = self.attention(
            query=link_features,
            key=link_features,
            value=link_features
        )
        
        # Residual Connection (Important for gradient flow)
        coordinated_features = link_features + attn_out # (seq_len, num_links, hidden_size)
        # Apply layer normalization
        # coordinated_features = self.layer_norm(coordinated_features)

        # --- Phase 3: Global Aggregation ---
        # Now that every link vector contains info about the global state (thanks to attention),
        # we can safely average them to get the "System Representation".
        
        # (seq_len, num_links, hidden) -> (seq_len, hidden)
        global_state = coordinated_features.mean(dim=1) 
        
        # --- Phase 4: Value Estimation ---
        value = self.value_head(F.elu(global_state)) # (seq_len, 1)

        return value, hidden_out

def train_on_policy_multi_agent(env, agents, delta_actions=False, num_episodes=50,
                                randomize=False,
                                agents_saved_dir: str = None, use_wandb: bool = True,
                                val_freq: int = 10, num_val_episodes: int = 3):
    """
    Train multiple on-policy agents (PPO) in a multi-agent environment.

    Args:
        env: PettingZoo ParallelEnv
        agents: Dict mapping agent_id -> PPOAgent
        delta_actions: If True, agents output delta actions
        num_episodes: Total number of episodes to train
        randomize: If True, randomize environment at reset
        agents_saved_dir: Directory to save agent checkpoints
        use_wandb: If True, log metrics to wandb (default: True)
        val_freq: Validation frequency - run validation every N episodes (default: 10)
        num_val_episodes: Number of validation episodes to run (default: 3)

    Returns:
        return_dict: Dict mapping agent_id -> list of episode returns
    """
    # Initialize wandb if available and not already initialized
    if use_wandb and WANDB_AVAILABLE:
        if wandb.run is None:
            wandb.init(project="crowd-control-rl", name="ppo-training")
    
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
                if i_episode == 0:
                    obs, infos = env.reset(options={'randomize': False})
                else:
                    obs, infos = env.reset(options={'randomize': randomize})

                # Initialize state history queues for stacked observations
                state_stack = {}
                if uses_stacked_obs:
                    for agent_id in state_history_queue.keys():
                        state_history_queue[agent_id].clear()
                        for _ in range(agents[agent_id].stack_size):
                            state_history_queue[agent_id].append(obs[agent_id])
                        state_stack[agent_id] = np.array(state_history_queue[agent_id])

                episode_returns = {agent_id: 0.0 for agent_id in agents.keys()}
                episode_true_returns = {agent_id: 0.0 for agent_id in agents.keys()}  # Track true (un-normalized) rewards
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
                        # Track true (un-normalized) rewards if available
                        if agent_id in infos and 'true_reward' in infos[agent_id]:
                            episode_true_returns[agent_id] += infos[agent_id]['true_reward']
                        else:
                            episode_true_returns[agent_id] += rewards[agent_id]


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

                # Log rewards to wandb after each update step
                if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
                    log_dict = {
                        'episode': global_episode,
                        'avg_normalized_return': np.mean(list(episode_returns.values())),
                        'avg_true_return': np.mean(list(episode_true_returns.values())),
                        'total_normalized_return': sum(episode_returns.values()),
                        'total_true_return': sum(episode_true_returns.values()),
                        'episode_steps': step
                    }
                    # Log per-agent rewards
                    for agent_id in agents.keys():
                        log_dict[f'agent_{agent_id}_normalized_return'] = episode_returns[agent_id]
                        log_dict[f'agent_{agent_id}_true_return'] = episode_true_returns[agent_id]
                    wandb.log(log_dict)

                # Run validation and save best model (after half of training, every val_freq episodes)
                if agents_saved_dir and global_episode > num_episodes/2 and global_episode % val_freq == 0:
                    best_avg_return = validate_and_save_best(
                        env, agents, agents_saved_dir,
                        delta_actions=delta_actions,
                        num_val_episodes=num_val_episodes,
                        randomize=True, # always randomize during validation
                        best_avg_return=best_avg_return,
                        global_episode=global_episode,
                        use_wandb=use_wandb and WANDB_AVAILABLE
                    )
                    # best_avg_return = save_with_best_return(agents, agents_saved_dir, episode_returns=episode_returns, best_avg_return=best_avg_return, global_episode=global_episode)
                # Update progress bar with both normalized and true returns
                if (i_episode+1) % 10 == 0:
                    avg_return = np.mean([np.mean(return_dict[aid][-10:]) for aid in agents.keys()])
                    avg_true_return = np.mean(list(episode_true_returns.values()))
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes/10 * i + i_episode+1),
                        'norm_ret': '%.3f' % avg_return,
                        'true_ret': '%.3f' % avg_true_return,
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
                 use_gat_lstm=False, gat_hidden_size=64, gat_num_heads=4,
                 use_param_noise=False, param_noise_std=0.1, param_noise_decay=0.995,
                 param_noise_std_min=0.01):
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
            use_param_noise: If True, apply parameter noise to actor for exploration
            param_noise_std: Initial standard deviation for parameter noise
            param_noise_decay: Decay factor for param_noise_std per episode (default 0.995)
            param_noise_std_min: Minimum param_noise_std (default 0.01)
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
        self.use_gat_lstm = use_gat_lstm

        if use_gat_lstm:
            # GAT-LSTM settings
            if self.obs_dim % self.act_dim != 0:
                raise ValueError("obs_dim must be divisible by act_dim when use_gat_lstm=True")
            # self.features_per_link = self.obs_dim // self.act_dim
            self.lstm_hidden_size = lstm_hidden_size
            self.gat_hidden_size = gat_hidden_size
            self.gat_num_heads = gat_num_heads
            self.num_lstm_layers = num_lstm_layers

            # Create GAT-LSTM networks
            self.actor = GATLSTMPolicyNetwork(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                lstm_hidden_size=lstm_hidden_size,
                gat_hidden_size=gat_hidden_size,
                num_heads=gat_num_heads,
                num_lstm_layers=num_lstm_layers
            )
            self.value_net = GATLSTMValueNetwork(
                obs_dim=self.obs_dim,
                num_links=self.act_dim,
                lstm_hidden_size=lstm_hidden_size,
                gat_hidden_size=gat_hidden_size,
                num_heads=gat_num_heads,
                num_lstm_layers=num_lstm_layers
            )
            # Hidden states managed internally in GAT-LSTM networks
            self.actor_hidden = None
            self.critic_hidden = None

        elif use_stacked_obs:
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
            # self.actor = UDLSTMPolicyNetwork(obs_dim, act_dim, hidden_size=lstm_hidden_size,
            #                               num_layers=num_lstm_layers)
            # self.value_net = UDLSTMValueNetwork(obs_dim, act_dim, hidden_size=lstm_hidden_size,
            #                                  num_layers=num_lstm_layers)
            # self.actor = LSTMPolicyNetwork(obs_dim, act_dim, hidden_size=lstm_hidden_size,
            #                               num_layers=num_lstm_layers)
            # self.value_net = LSTMValueNetwork(obs_dim, act_dim, hidden_size=lstm_hidden_size,
            #                                num_layers=num_lstm_layers)
            self.actor = AttentionPolicy(obs_dim, act_dim, hidden_size=lstm_hidden_size,
                                          num_layers=num_lstm_layers)
            self.value_net = AttentionValueNetwork(obs_dim, act_dim, hidden_size=lstm_hidden_size,
                                                   num_layers=num_lstm_layers)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=critic_lr)
        self.device = device

        # Parameter noise settings for exploration
        self.use_param_noise = use_param_noise
        self.param_noise_std_initial = param_noise_std
        self.param_noise_std = param_noise_std
        self.param_noise_decay = param_noise_decay
        self.param_noise_std_min = param_noise_std_min
        self._param_noise_applied = False  # Track if noise is currently applied
        self._original_actor_params = None  # Store original params when noise is applied

    def reset_buffer(self):
        """Clear rollout buffer, reset LSTM hidden states, and apply parameter noise for new episode."""
        self.transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        # Reset LSTM hidden states (only if using LSTM or GAT-LSTM, will be initialized in first forward pass)
        if not self.use_stacked_obs:
            self.actor_hidden = None
            self.critic_hidden = None
        
        # Apply parameter noise at the start of each episode
        if self.use_param_noise:
            self._apply_param_noise()

    def _apply_param_noise(self):
        """Apply Gaussian noise to actor network parameters for exploration."""
        # First restore original params if noise was previously applied
        if self._param_noise_applied:
            self._restore_actor_params()
        
        # Store original parameters
        self._original_actor_params = {
            name: param.data.clone() for name, param in self.actor.named_parameters()
        }
        
        # Add noise to each parameter
        with torch.no_grad():
            for name, param in self.actor.named_parameters():
                if param.requires_grad:
                    noise = torch.randn_like(param) * self.param_noise_std
                    param.data.add_(noise)
        
        self._param_noise_applied = True

    def _restore_actor_params(self):
        """Restore original actor parameters (remove noise)."""
        if self._original_actor_params is not None:
            with torch.no_grad():
                for name, param in self.actor.named_parameters():
                    if name in self._original_actor_params:
                        param.data.copy_(self._original_actor_params[name])
            self._original_actor_params = None
        self._param_noise_applied = False

    def _decay_param_noise_std(self):
        """Apply decay to parameter noise standard deviation."""
        self.param_noise_std = max(
            self.param_noise_std_min,
            self.param_noise_std * self.param_noise_decay
        )

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
        if self.use_gat_lstm:
            # GAT-LSTM forward pass (graph structure handled internally)
            mu, sigma, self.actor_hidden = self.actor(state_tensor, adj=None, hidden=self.actor_hidden)
        elif self.use_stacked_obs:
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
        # Restore original actor parameters before update (remove noise)
        if self.use_param_noise and self._param_noise_applied:
            self._restore_actor_params()
        
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
        if self.use_stacked_obs or self.use_gat_lstm:
            pass
        else:
            # LSTM: states is (T, obs_dim)
            # Add batch dimension: (1, T, obs_dim)
            states_seq = states.unsqueeze(0)  # (1, T, obs_dim)
            next_states_seq = next_states.unsqueeze(0)  # (1, T, obs_dim)

        # Compute targets with no_grad - these are constants for the update
        with torch.no_grad():
            # Process entire sequences through value network
            if self.use_gat_lstm:
                # GAT-LSTM version: process each timestep separately
                next_values = self.value_net(next_states, adj=None)[0]  # (T, 1)
                current_values = self.value_net(states, adj=None)[0]  # (T, 1)
            elif self.use_stacked_obs:
                # For stacked, we need to process each timestep separately or reshape
                # StackedValueNetwork.forward_all_steps expects (batch, seq, stack_size, obs_dim)
                # But we have (1, T, stack_size, obs_dim), which is correct
                next_values = self.value_net(next_states)
                current_values = self.value_net(states)
            else:
                # LSTM version
                next_values, _ = self.value_net(next_states_seq)  # (1, T, 1)
                next_values = next_values.squeeze(0)  # (T, 1)

                current_values, _ = self.value_net(states_seq)  # (1, T, 1)
                current_values = current_values.squeeze(0)  # (T, 1)

            td_target = rewards + self.gamma * next_values * (1 - dones)
            td_delta = td_target - current_values

            # Use shared compute_gae function
            advantage = compute_gae(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

            # Normalize advantage (crucial for stable training)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            # Process sequence through actor to get old log probs
            if self.use_gat_lstm:
                mu, std, _ = self.actor(states, adj=None)  # (T, act_dim)
            elif self.use_stacked_obs:
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
            if self.use_gat_lstm:
                mu, std, _ = self.actor(states, adj=None)
            elif self.use_stacked_obs:
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
            if self.use_gat_lstm:
                current_values, _ = self.value_net(states, adj=None)
            elif self.use_stacked_obs:
                current_values = self.value_net(states)
            else:
                current_values, _ = self.value_net.forward(states_seq)
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

        #     KL early stopping
            with torch.no_grad():
                approx_kl = (log_probs - old_log_probs).mean()
                if approx_kl > 1.5 * self.kl_tolerance:
                    break

        # Decay entropy coefficient after update
        self._decay_entropy_coef()
        
        # Decay parameter noise std after update
        if self.use_param_noise:
            self._decay_param_noise_std()

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
            'use_gat_lstm': self.use_gat_lstm,
            'use_param_noise': self.use_param_noise,
            'param_noise_std': self.param_noise_std_initial,
            'param_noise_decay': self.param_noise_decay,
            'param_noise_std_min': self.param_noise_std_min,
        }

        if self.use_gat_lstm:
            config.update({
                'lstm_hidden_size': self.lstm_hidden_size,
                'gat_hidden_size': self.gat_hidden_size,
                'gat_num_heads': self.gat_num_heads,
                'num_lstm_layers': self.num_lstm_layers,
            })
        elif self.use_stacked_obs:
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
        # Ensure we save original (non-noisy) parameters
        if self.use_param_noise and self._param_noise_applied:
            self._restore_actor_params()
        
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.value_net.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.get_config(),
            'update_count': self.update_count,
            'current_entropy_coef': self.entropy_coef,
            'current_param_noise_std': self.param_noise_std if self.use_param_noise else None,
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
        # Restore param noise state if available
        if 'current_param_noise_std' in checkpoint and checkpoint['current_param_noise_std'] is not None:
            self.param_noise_std = checkpoint['current_param_noise_std']