# -*- coding: utf-8 -*-
# @Time    : 13/10/2025 15:45
# @Author  : mmai
# @FileName: rl_example
# @Software: PyCharm

"""
Example usage of PedNet RL environment with predefined controllers.

Shows how to configure and run the multi-agent environment with
specific separator and gater controller placements.
"""

import os
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import numpy as np
from src.rl import PedNetParallelEnv


def test_environment():
    """Test the PedNet environment with predefined controllers."""
    dataset = "nine_node"
    try:
        # Initialize environment
        env = PedNetParallelEnv(dataset)
        print(f"Created PedNet environment with dataset: {dataset}")
        
        # Reset environment
        observations, infos = env.reset(seed=42)
        print(f"Environment reset. Found {len(env.agents)} agents:")
        
        # Run a few simulation steps with random actions
        print("\nRunning simulation steps...")
        for step in range(5):
            # Generate random actions for all agents
            actions = {}
            for agent_id in env.agents:
                action_space = env.action_space(agent_id)
                actions[agent_id] = action_space.sample()
                
                # For gater agents, apply action mask
                if agent_id.startswith("gate:"):
                    action_mask = infos[agent_id].get("action_mask")
                    if action_mask is not None:
                        actions[agent_id] = actions[agent_id] * action_mask
            
            # Step environment
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Print step results
            total_reward = sum(rewards.values())
            any_done = any(terminations.values()) or any(truncations.values())
            print(f"  Step {step + 1}: total_reward={total_reward:.3f}, done={any_done}")
            
            if any_done:
                print("  Episode terminated.")
                break
        
        print("Environment test completed successfully!")
        
    except Exception as e:
        print(f"Error testing environment: {e}")
        print("Make sure your network has the required Separator links and nodes.")
        print("Check the controller configuration matches your network topology.")
        raise


if __name__ == "__main__":
    test_environment()
