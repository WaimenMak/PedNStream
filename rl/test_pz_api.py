# -*- coding: utf-8 -*-
"""
Test PettingZoo API compliance for PedNetParallelEnv.

This script verifies that the environment adheres to PettingZoo's parallel API
standards, ensuring compatibility with multi-agent RL frameworks.
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from pettingzoo.test import parallel_api_test
from rl import PedNetParallelEnv


def test_pettingzoo_compliance():
    """Run PettingZoo parallel API compliance test."""
    print("=" * 60)
    print("Testing PettingZoo API Compliance")
    print("=" * 60)
    
    dataset = "nine_intersections"  # Simple scenario for testing
    
    try:
        print(f"\nInitializing environment with dataset: {dataset}")
        env = PedNetParallelEnv(
            dataset=dataset,
            normalize_obs=True,
            with_density_obs=True
        )
        
        print(f"Found {len(env.possible_agents)} agents:")
        for agent_id in env.possible_agents:
            agent_type = env.agent_manager.get_agent_type(agent_id)
            print(f"  - {agent_id} ({agent_type})")
        
        print("\nRunning PettingZoo parallel API test...")
        print("This will test reset, step, action/observation spaces, etc.")
        print("-" * 60)
        
        # Run the official PettingZoo compliance test
        # This will raise an error if the API is not compliant
        parallel_api_test(env, num_cycles=100)
        
        print("-" * 60)
        print("\n✅ SUCCESS! Environment is PettingZoo API compliant.")
        print("\nThe environment passed all checks:")
        print("  ✓ Reset returns (observations, infos)")
        print("  ✓ Step returns (obs, rewards, terms, truncs, infos)")
        print("  ✓ Action/observation spaces are valid")
        print("  ✓ All agents receive consistent data")
        print("  ✓ Termination logic is correct")
        
        return True
        
    except Exception as e:
        print("\n❌ FAILED! Environment has API compliance issues.")
        print(f"\nError: {e}")
        print("\nPlease fix the issues above before training RL agents.")
        return False


if __name__ == "__main__":
    success = test_pettingzoo_compliance()
    sys.exit(0 if success else 1)

