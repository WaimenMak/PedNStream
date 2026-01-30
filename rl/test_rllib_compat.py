# -*- coding: utf-8 -*-
"""
Test RLlib compatibility with PedNetParallelEnv.

This script verifies that the PettingZoo environment can be successfully
wrapped and used with Ray RLlib for multi-agent reinforcement learning.
"""

import sys
from pathlib import Path

from ray.tune.examples.pbt_dcgan_mnist.common import batch_size

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import numpy as np
from rl import PedNetParallelEnv

# RLlib imports
try:
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    from ray.rllib.algorithms.ppo import PPOConfig
    import ray
    RLLIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RLlib not available: {e}")
    RLLIB_AVAILABLE = False


def test_basic_env_instantiation():
    """Test 1: Basic environment instantiation."""
    print("\n" + "=" * 70)
    print("Test 1: Basic Environment Instantiation")
    print("=" * 70)
    
    try:
        env = PedNetParallelEnv(
            dataset="nine_intersections",
            normalize_obs=True,
            with_density_obs=True
        )
        
        print(f"✓ Environment created successfully")
        print(f"  - Dataset: nine_intersections")
        print(f"  - Agents: {len(env.possible_agents)}")
        print(f"  - Agent IDs: {env.possible_agents}")
        
        # Check action and observation spaces
        for agent_id in env.possible_agents[:2]:  # Check first 2 agents
            obs_space = env.observation_space(agent_id)
            act_space = env.action_space(agent_id)
            print(f"  - {agent_id}:")
            print(f"    * Obs space: {obs_space}")
            print(f"    * Act space: {act_space}")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reset_and_step():
    """Test 2: Reset and step functionality."""
    print("\n" + "=" * 70)
    print("Test 2: Reset and Step Functionality")
    print("=" * 70)
    
    try:
        env = PedNetParallelEnv(
            dataset="nine_intersections",
            normalize_obs=True,
            with_density_obs=True
        )
        
        # Test reset
        observations, infos = env.reset()
        print(f"✓ Reset successful")
        print(f"  - Observations received for {len(observations)} agents")
        print(f"  - Info keys: {list(infos[env.possible_agents[0]].keys())}")
        
        # Sample random actions
        actions = {
            agent_id: env.action_space(agent_id).sample()
            for agent_id in env.possible_agents
        }
        
        # Test step
        obs, rewards, terms, truncs, infos = env.step(actions)
        print(f"✓ Step successful")
        print(f"  - Observations: {len(obs)}")
        print(f"  - Rewards: {len(rewards)}")
        print(f"  - Sample reward: {list(rewards.values())[0]:.4f}")
        print(f"  - Terminations: {any(terms.values())}")
        print(f"  - Truncations: {any(truncs.values())}")
        
        # Verify return types
        assert isinstance(obs, dict), "Observations must be dict"
        assert isinstance(rewards, dict), "Rewards must be dict"
        assert isinstance(terms, dict), "Terminations must be dict"
        assert isinstance(truncs, dict), "Truncations must be dict"
        assert isinstance(infos, dict), "Infos must be dict"
        
        # Verify all agents present
        assert set(obs.keys()) == set(env.possible_agents), "All agents must have observations"
        assert set(rewards.keys()) == set(env.possible_agents), "All agents must have rewards"
        
        print(f"✓ Return format validation passed")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rllib_wrapper():
    """Test 3: RLlib PettingZoo wrapper compatibility."""
    print("\n" + "=" * 70)
    print("Test 3: RLlib PettingZoo Wrapper")
    print("=" * 70)
    
    if not RLLIB_AVAILABLE:
        print("✗ Skipped: RLlib not installed")
        print("  Install with: pip install 'ray[rllib]'")
        return False
    
    try:
        # Create base environment
        base_env = PedNetParallelEnv(
            dataset="nine_intersections",
            normalize_obs=True,
            with_density_obs=True
        )
        
        # Wrap with RLlib wrapper
        env = ParallelPettingZooEnv(base_env)
        print(f"✓ RLlib wrapper created successfully")
        
        # Test reset
        obs_dict, info_dict = env.reset()
        print(f"✓ Wrapped environment reset successful")
        print(f"  - Observations shape: {type(obs_dict)}")
        
        # Test step with random actions
        actions = {
            agent_id: env.action_space[agent_id].sample()
            for agent_id in env.get_agent_ids()
        }
        obs, rewards, dones, truncated, infos = env.step(actions)
        print(f"✓ Wrapped environment step successful")
        print(f"  - Step returned {len(obs)} observations")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rllib_config():
    """Test 4: RLlib algorithm configuration."""
    print("\n" + "=" * 70)
    print("Test 4: RLlib Algorithm Configuration")
    print("=" * 70)
    
    if not RLLIB_AVAILABLE:
        print("✗ Skipped: RLlib not installed")
        return False
    
    try:
        # Initialize Ray (if not already initialized)
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=2, log_to_driver=False)
            print(f"✓ Ray initialized")
        
        # Create a wrapper class for RLlib (RLlib expects a class, not a function)
        class PedNetRLlibEnv(ParallelPettingZooEnv):
            def __init__(self, config=None):
                config = config or {}
                base_env = PedNetParallelEnv(
                    dataset=config.get("dataset", "nine_intersections"),
                    normalize_obs=config.get("normalize_obs", True),
                    with_density_obs=config.get("with_density_obs", True)
                )
                super().__init__(base_env)
        
        # Test environment creation
        test_env = PedNetRLlibEnv({"dataset": "nine_intersections"})
        agent_ids = test_env.get_agent_ids()
        print(f"✓ Environment class created")
        print(f"  - Agents: {agent_ids}")
        
        # Create minimal PPO config using the environment class
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
                num_env_runners=0,  # Use only local worker for testing
                rollout_fragment_length=50
            )
            .training(
                train_batch_size=100,
                minibatch_size=40,
                # batch_size=32
            )
        )
        
        print(f"✓ PPO config created successfully")
        print(f"  - Framework: torch")
        print(f"  - Multi-agent policies: {len(agent_ids)}")
        print(f"  - Rollout workers: 0 (local only)")
        
        # Try to build the algorithm (but don't train)
        algo = config.build()
        print(f"✓ PPO algorithm built successfully")
        
        # Try one training iteration
        print(f"  Running one training iteration (this may take a minute)...")
        result = algo.train()
        print(f"✓ Training iteration completed")
        print(f"  - Episodes: {result.get('episodes_this_iter', 'N/A')}")
        print(f"  - Timesteps: {result.get('timesteps_this_iter', 'N/A')}")
        
        # Cleanup
        algo.stop()
        print(f"✓ Algorithm cleanup successful")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if ray.is_initialized():
            ray.shutdown()
            print("Ray shutdown")


def test_multi_episode_rollout():
    """Test 5: Multi-episode rollout with RLlib wrapper."""
    print("\n" + "=" * 70)
    print("Test 5: Multi-Episode Rollout")
    print("=" * 70)
    
    if not RLLIB_AVAILABLE:
        print("✗ Skipped: RLlib not installed")
        return False
    
    try:
        # Create wrapped environment
        base_env = PedNetParallelEnv(
            dataset="nine_intersections",
            normalize_obs=True,
            with_density_obs=True
        )
        env = ParallelPettingZooEnv(base_env)
        
        num_episodes = 3
        num_steps_per_episode = 10
        
        print(f"Running {num_episodes} episodes with {num_steps_per_episode} steps each")
        
        for episode in range(num_episodes):
            obs, infos = env.reset()
            episode_rewards = {agent_id: 0.0 for agent_id in env.get_agent_ids()}
            
            for step in range(num_steps_per_episode):
                # Random actions
                actions = {
                    agent_id: env.action_space[agent_id].sample()
                    for agent_id in env.get_agent_ids()
                }
                
                obs, rewards, dones, truncated, infos = env.step(actions)
                
                # Accumulate rewards
                for agent_id, reward in rewards.items():
                    episode_rewards[agent_id] += reward
                
                # Check if episode ended early
                if dones.get("__all__", False):
                    print(f"  Episode {episode + 1} ended at step {step + 1}")
                    break
            
            avg_reward = np.mean(list(episode_rewards.values()))
            print(f"  Episode {episode + 1}: avg_reward={avg_reward:.4f}")
        
        print(f"✓ Multi-episode rollout completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all RLlib compatibility tests."""
    print("\n" + "=" * 70)
    print("RLLIB COMPATIBILITY TEST SUITE")
    print("=" * 70)
    print("\nTesting PedNetParallelEnv compatibility with Ray RLlib")
    
    if not RLLIB_AVAILABLE:
        print("\n⚠️  WARNING: RLlib is not installed!")
        print("Install with: pip install 'ray[rllib]>=2.9.0'")
        print("\nRunning basic tests only...\n")
    
    results = {}
    
    # Run tests
    results['instantiation'] = test_basic_env_instantiation()
    results['reset_step'] = test_reset_and_step()
    results['wrapper'] = test_rllib_wrapper()
    results['config'] = test_rllib_config()
    results['rollout'] = test_multi_episode_rollout()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10s} | {test_name}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print("-" * 70)
    print(f"Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 SUCCESS! Environment is fully compatible with RLlib!")
        print("\nYou can now use this environment with RLlib algorithms:")
        print("  - PPO, SAC, DQN, etc.")
        print("  - Multi-agent training with independent or shared policies")
        print("  - Distributed training with multiple workers")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
    
    print("=" * 70)
    
    return all(results.values())


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

