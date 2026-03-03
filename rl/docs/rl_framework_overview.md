# RL Framework for Pedestrian Crowd Control

## Overview

This project applies reinforcement learning to control pedestrian traffic flow in bidirectional corridor networks. The simulation is powered by the **PedNStream** LTM (Link Transmission Model), and the RL layer is built on top of the [PettingZoo](https://pettingzoo.farama.org/) multi-agent API.

## Environment

### `PedNetParallelEnv` (`rl/pz_pednet_env.py`)

A PettingZoo `ParallelEnv` wrapping the PedNStream network simulation.

| Property | Details |
|----------|---------|
| **Simulator** | PedNStream LTM (link-level pedestrian flow model) |
| **Time step** | Each `env.step()` advances the simulation by one LTM time step |
| **Episode length** | `simulation_steps` defined by the network dataset |
| **Action gap** | Configurable (`action_gap`): actions are applied every N simulation steps |
| **Late start** | Configurable probability of starting the agent at a later time step for training diversity |

### Agent Types

Two agent types are auto-discovered from the network configuration via `AgentManager`:

| Agent Type | ID Format | Controls | Example |
|------------|-----------|----------|---------|
| **Gater** | `gate_N` | `back_gate_width` on outgoing links of a node | Throttle flow entering/leaving a node |
| **Separator** | `sep_U_V` | `separator_width` on bidirectional corridors | Allocate corridor width between directions |

Currently, the primary focus is on **gater agents**. Each gater controls the gate widths of all outgoing links at its assigned node.

### Action Space (Gater)

- **Dimension**: `2 * num_outgoing_links` per agent (one action per direction per physical link)
- **Layout**: `[link0_back_gate, link0_rev_back_gate, link1_back_gate, link1_rev_back_gate, ...]`
- **Mode**: Delta actions — the agent outputs `Δwidth`, clipped to `±max_delta` (default 2.5), applied on top of the current gate width
- **Physical bounds**: Gate width is clipped to `[0, link.width]`

### Observation Space (Gater)

Features are extracted per **physical link** (outgoing links of the node). Multiple observation modes are supported:

| Mode | Features/Link | Feature Layout |
|------|---------------|----------------|
| `option1` | 4 | `[fwd_in, rev_out, back_gate, rev_front_gate]` |
| `option2` | 5 | `[fwd_in, rev_out, density, back_gate, rev_front_gate]` |
| `option3` | 6 | `[fwd_in, fwd_out, rev_in, rev_out, back_gate, rev_back_gate]` |
| `option4` | 8 | `[fwd_in, fwd_out, fwd_density, rev_in, rev_out, rev_density, back_gate, rev_back_gate]` |
| `option5` | 6 | `[tt_ratio, density_ratio, demand, throughput, back_gate, rev_back_gate]` |

**Key pattern for option3/option4**: Features are grouped as `[fwd_flow_features..., rev_flow_features..., fwd_gate, rev_gate]`. This layout is exploited by the directional token design (see `rl/docs/directional_token_design.md`).

### Reward Function

Per-agent reward for gater agents is computed per time step as:

```
reward = w1 * flow_term + w2 * tt_term + w3 * fairness_term + w4 * demand_term
```

| Component | Weight | Description |
|-----------|--------|-------------|
| `flow_term` | 1.0 | Normalized throughput (outflow / capacity) per link, summed |
| `tt_term` | 1.0 | Negative normalized travel time penalty: `-clip(log(T/T_free), 0, 1)` |
| `fairness_term` | 0.5 | Penalizes density imbalance across links (only when max density > critical) |
| `demand_term` | 2.0 | Normalized demand (reverse inflow / capacity) |

### Normalization Wrapper

`RunningNormalizeWrapper` (`rl/rl_utils.py`) provides:
- **Observation normalization**: Running mean/std, excluding gate width features
- **Reward normalization**: Running std of returns (preserves sign)
- **Training/eval mode**: Stats are frozen during validation

## RL Algorithms

### PPO-HRL (`rl/agents/PPO_hrl.py`)

Single-agent PPO with **Hierarchical Temporal Abstraction**:

- At each **decision point**, the agent outputs:
  - **Continuous action** `δ_width` (one per gate direction) — sampled from `N(μ, σ)`
  - **Discrete duration** `k ∈ {1, ..., max_duration}` — sampled from `Categorical(logits)`
- The action is **held for k env steps** (macro-transition), accumulating discounted reward
- PPO update treats each macro-transition as a single step
- **Variable-gamma GAE**: discount between decisions `i` and `i+1` is `γ^{k_i}`

#### Network Architecture: `DurationAttentionPolicy`

```
Input obs (obs_dim)
    │
    ▼
Reshape to physical links → Directional token construction (fwd_idx/rev_idx)
    │                         → 2 tokens per physical link, each seeing all features
    ▼
Shared LSTM (per-token, across time)
    │
    ▼
Link projection (Linear)
    │
    ▼
Multi-Head Self-Attention (inter-direction, inter-link coordination)
    │
    ▼
LayerNorm (residual connection)
    │
    ├──→ Shared mean_head → 1 action per token → interleaved output (act_dim)
    ├──→ Shared std_head  → 1 std per token → interleaved output (act_dim)
    └──→ Global mean-pool → duration_fc → duration logits (max_duration)
```

See `rl/docs/directional_token_design.md` for details on the directional token construction.

#### Value Network: `DurationAttentionValueNetwork`

Same LSTM + Attention backbone. Supports multiple fusion strategies for global pooling:
- `attention`: Learned query attends to all link tokens (recommended)
- `mean`: Simple mean pooling
- `max`: Max pooling
- `mean_max`: Concatenated mean + max
- `gated`: Learned importance weights per link

### MAPPO-HRL (`rl/marl/MAPPO_hrl.py`)

Multi-Agent PPO extension for environments with multiple gater agents:

| Aspect | PPO-HRL | MAPPO-HRL |
|--------|---------|-----------|
| Actor | Per-agent (local obs) | Per-agent (local obs) |
| Critic | Per-agent (local obs) | Per-agent (local obs) or Shared (global state) |
| Reward | Per-agent | Global (sum of all agents' rewards) |
| Duration | Per-agent | Per-agent (asynchronous decisions) |

**Shared critic mode**: The centralized critic receives the concatenated observations of all agents. Tokens = `num_agents × num_links_per_agent`, each with `feat_per_link` features.

**Per-agent critic mode**: Each agent has its own local critic (`num_agents=1` in the value network).

#### Training Loop (`rl/marl/train_mappo.py`)

- Asynchronous macro-transitions: each agent independently decides its duration
- Global reward: `sum(rewards.values())` at each env step
- Cumulative rewards are temporally discounted within each macro-transition
- Batch training: `num_trajectories_per_update` episodes collected before PPO update
- Shared critic updated via `update_shared_critic()`, then each agent updates its actor via `update_actor_only()`

## Training Infrastructure

### Training Scripts

| Script | Algorithm | Description |
|--------|-----------|-------------|
| `rl/train_rl.py` | PPO-HRL | Single training loop for one dataset |
| `rl/marl/train_mappo.py` | MAPPO-HRL | Multi-agent training with optional shared critic |

### Key Hyperparameters

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| `SEED` | varies | Random seed for reproducibility |
| `STATE_OPTION` | `"option3"` | Observation mode |
| `hidden_size` | 64 | LSTM and attention hidden dimension |
| `num_heads` | 2 | Multi-head attention heads |
| `max_duration` | 7 | Maximum macro-action duration |
| `max_delta` | 2.5 | Maximum gate width change per decision |
| `gamma` | 0.99 | Discount factor |
| `lmbda` | 0.95-0.96 | GAE lambda |
| `tm_window` | 20 | TBPTT chunk size |
| `num_trajectories_per_update` | 2-4 | Batch size (trajectories) |
| `duration_entropy_coef` | 0.05 | Duration head entropy bonus |
| `entropy_coef` | 0.01-0.04 | Action head entropy bonus |

### Validation & Model Selection

- `validate_and_save_best()` creates a **fresh environment** for validation (no state leakage)
- Models are saved based on **average return across all agents** (total reward, not per-agent)
- Validation uses **deterministic** action selection with duration commitment replay
- Normalization statistics are copied but frozen during validation

### Datasets

Networks are loaded from the PedNStream dataset system. Key scenarios:

| Dataset | Description |
|---------|-------------|
| `butterfly_scC/D/F/G` | Butterfly network variants with different demand patterns |
| `one_intersection_v0` | Single intersection for debugging |
| `small_network` | Small test network |
| `nine_intersections` | Larger multi-agent scenario |

## File Structure

```
rl/
├── pz_pednet_env.py          # PettingZoo environment wrapper
├── discovery.py              # AgentManager: auto-discovers agents from network
├── spaces.py                 # SpaceBuilder: action/observation space definitions
├── builders.py               # ObservationBuilder + ActionApplier
├── rl_utils.py               # Normalization, save/load, validation utilities
├── train_rl.py               # PPO-HRL training script
├── agents/
│   └── PPO_hrl.py            # PPOAgentHRL + networks (DurationAttentionPolicy, etc.)
├── marl/
│   ├── MAPPO_hrl.py          # MAPPOAgentHRL + MAPPO-specific networks
│   └── train_mappo.py        # MAPPO training script
└── docs/
    ├── directional_token_design.md  # Detailed design of bidirectional tokenization
    └── rl_framework_overview.md     # This file
```
