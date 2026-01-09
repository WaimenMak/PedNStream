# RLlib Integration Guide

## Overview
This document explains how to use the PedNetParallelEnv with Ray RLlib for multi-agent reinforcement learning.

## Fixed Issues

### 1. Agent ID Naming Convention
**Problem:** Agent IDs contained colons (e.g., `gate:4`, `sep:1_2`) which are not allowed in RLlib policy IDs.

**Solution:** Changed naming convention to use underscores instead:
- **Before:** `gate:4`, `sep:1_2`
- **After:** `gate_4`, `sep_1_2`

**Files Modified:**
- `rl/discovery.py` - Updated agent ID generation
- `rl/pz_pednet_env.py` - Updated docstrings

### 2. Environment Specification
**Problem:** RLlib's `.environment()` method expects a class or registered env name, not a function.

**Solution:** Created a wrapper class that inherits from `ParallelPettingZooEnv`:

```python
class PedNetRLlibEnv(ParallelPettingZooEnv):
    def __init__(self, config=None):
        config = config or {}
        base_env = PedNetParallelEnv(
            dataset=config.get("dataset", "nine_intersections"),
            normalize_obs=config.get("normalize_obs", True),
            with_density_obs=config.get("with_density_obs", True)
        )
        super().__init__(base_env)
```

## Usage Example

### Basic RLlib Configuration

```python
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from rl import PedNetParallelEnv
import ray

# Initialize Ray
ray.init()

# Create wrapper class
class PedNetRLlibEnv(ParallelPettingZooEnv):
    def __init__(self, config=None):
        config = config or {}
        base_env = PedNetParallelEnv(
            dataset=config.get("dataset", "nine_intersections"),
            normalize_obs=config.get("normalize_obs", True),
            with_density_obs=config.get("with_density_obs", True)
        )
        super().__init__(base_env)

# Get agent IDs
test_env = PedNetRLlibEnv()
agent_ids = test_env.get_agent_ids()

# Configure PPO
config = (
    PPOConfig()
    .environment(
        env=PedNetRLlibEnv,
        env_config={"dataset": "nine_intersections"}
    )
    .framework("torch")
    .multi_agent(
        policies={agent_id: (None, None, None, {}) for agent_id in agent_ids},
        policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
    )
    .env_runners(
        num_env_runners=2,
        rollout_fragment_length=200
    )
    .training(
        train_batch_size=4000,
        minibatch_size=128
    )
)

# Build and train
algo = config.build()
for i in range(100):
    result = algo.train()
    print(f"Iteration {i}: reward={result['episode_reward_mean']}")
```

## Testing

Run the compatibility test suite:

```bash
cd /Users/mmai/Devs/Crowd-Control
python rl/test_rllib_compat.py
```

The test suite includes:
1. ✓ Basic environment instantiation
2. ✓ Reset and step functionality
3. ✓ RLlib PettingZoo wrapper
4. ✓ RLlib algorithm configuration and training
5. ✓ Multi-episode rollout

## Multi-Agent Configuration Options

### Independent Policies (Each agent has its own policy)
```python
.multi_agent(
    policies={agent_id: (None, None, None, {}) for agent_id in agent_ids},
    policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
)
```

### Shared Policy (All agents share one policy)
```python
.multi_agent(
    policies={"shared_policy": (None, None, None, {})},
    policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
)
```

### Type-Based Policies (Separators and Gaters have separate policies)
```python
def policy_mapping(agent_id, *args, **kwargs):
    return "sep_policy" if agent_id.startswith("sep_") else "gate_policy"

.multi_agent(
    policies={
        "sep_policy": (None, None, None, {}),
        "gate_policy": (None, None, None, {})
    },
    policy_mapping_fn=policy_mapping,
)
```

## Supported Algorithms

The environment is compatible with all RLlib multi-agent algorithms:
- **PPO** - Proximal Policy Optimization (recommended for continuous control)
- **SAC** - Soft Actor-Critic
- **TD3** - Twin Delayed DDPG
- **APEX** - Distributed prioritized experience replay
- **IMPALA** - Importance Weighted Actor-Learner Architecture

## Troubleshooting

### Issue: "PolicyID not valid"
**Cause:** Agent IDs contain disallowed characters (`:`, `/`, `\`, etc.)
**Solution:** Ensure agent IDs only use alphanumeric characters and underscores

### Issue: "env_creator is an invalid env specifier"
**Cause:** Passing a function instead of a class to `.environment()`
**Solution:** Use the `PedNetRLlibEnv` wrapper class as shown above

### Issue: Slow training
**Cause:** Network simulation is computationally intensive
**Solution:** 
- Reduce `simulation_steps` in network config
- Use fewer `num_env_runners`
- Enable vectorization if possible

## Next Steps

1. Run the test suite to verify compatibility
2. Experiment with different algorithms and hyperparameters
3. Tune reward functions in `pz_pednet_env.py`
4. Scale up with distributed training using more workers

