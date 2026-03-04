# Directional Token Design for Bidirectional Traffic Control

## Problem

In bidirectional pedestrian traffic scenarios, each physical link carries traffic in **two directions** (forward and reverse). The RL agent must control a gate for each direction independently (`back_gate_width` for forward, `reverse_link.back_gate_width` for reverse).

A standard MLP or naive token-per-physical-link architecture struggles because:
1. Forward and reverse flows physically conflict (shared corridor space).
2. The network must learn the same control rules twice for different parts of its input.
3. The action output couples two directions without explicit separation.

## Observation Layout (option3, builders.py)

Each physical link produces **6 features** in this order:

```
Index:  [0,         1,          2,         3,           4,            5]
Feature: fwd_in,    fwd_out,    rev_in,    rev_out,     back_gate,    rev_back_gate
Dir:     forward    forward     reverse    reverse      forward       reverse
```

For `N` physical links → `obs_dim = N * 6`, `act_dim = N * 2`.

## Action Layout (builders.py ActionApplier)

```
action[i*2]   → out_links[i].back_gate_width          (forward direction)
action[i*2+1] → out_links[i].reverse_link.back_gate_width  (reverse direction)
```

## Chosen Architecture: Index-Based Directional Reordering

### Core Idea

Instead of naively splitting the 6-feature block, we create **2 tokens per physical link** (10 tokens for 5 links), each seeing **all 6 features** but reordered so "my direction" always occupies positions `[0,1,2]` and "other direction" occupies `[3,4,5]`.

### Index Computation

```python
fppl = 6  # features per physical link
half = 3  # features per direction

fwd_idx = [0, 1, 4, 2, 3, 5]  # → [my_in, my_out, my_gate, other_in, other_out, other_gate]
rev_idx = [2, 3, 5, 0, 1, 4]  # → [my_in, my_out, my_gate, other_in, other_out, other_gate]
```

### Token Construction (forward pass)

```python
x_phys = x.view(seq_len, num_physical_links, 6)        # Group by physical link
fwd_tokens = x_phys[:, :, fwd_idx]                      # Forward perspective
rev_tokens = x_phys[:, :, rev_idx]                       # Reverse perspective
x_links = torch.stack([fwd_tokens, rev_tokens], dim=2)   # (seq_len, N, 2, 6)
x_links = x_links.view(seq_len, num_links, 6)            # (seq_len, 2N, 6)
# Token order: [fwd₀, rev₀, fwd₁, rev₁, ...] matches action layout
```

### Architecture Summary

| Component | Details |
|-----------|---------|
| **Tokens** | `2N` directional tokens, each with 6 features |
| **LSTM** | Shared weights, processes each token independently across time |
| **Attention** | `2N × 2N` multi-head attention (fwd of link A can attend to rev of link B) |
| **Action Head** | Single shared `mean_head: Linear(hidden, 1)` per token |
| **Duration Head** | Global mean-pool → duration logits |

### Why Single Shared Head Works

Because every token has the **same semantic layout** (`[my_in, my_out, my_gate, other_in, other_out, other_gate]`), the shared `mean_head` learns one universal, direction-agnostic rule. This weight sharing acts as a **regularizer**, forcing the network to learn traffic physics rather than direction-specific quirks.

## Approaches Tested and Results

| Approach | Description | Result |
|----------|-------------|--------|
| Naive split (old `act_dim//2` links, 2 actions per head) | Each physical link = 1 token, `mean_head → 2` | Baseline, works for unidirectional |
| Naive split (`act_dim` links, 1 action per head) | 6 features split into 2 chunks of 3 (asymmetric) | Higher reward than index-based (cross-dir info within token) |
| Contiguous directional layout | Reorder obs to `[fwd_in, fwd_out, gate, rev_in, rev_out, rev_gate]` | Worse (lost cross-directional context per token) |
| **Index-based reordering (chosen)** | Both tokens see all 6 features, "my direction first" | **Best performance** |
| Dual-head (fwd head + rev head) | 1 token per physical link, separate mean heads | Not as good as shared head with reordering |

## Key Insight

Cross-directional information within each token is critical for bidirectional traffic. The network must compare forward vs reverse flow **within its immediate input** to make good gate decisions. The index-based reordering preserves this while giving every token a consistent semantic structure.

## Files Affected

- **`rl/agents/PPO_hrl.py`**: `DurationAttentionPolicy`, `DurationAttentionValueNetwork` — index buffers and reshaping
- **`rl/builders.py`**: Observation layout (option3) — unchanged, original order preserved
- **`rl/builders.py`**: `ActionApplier._apply_gater_action` — unchanged, interleaved `[fwd, rev, fwd, rev, ...]`
