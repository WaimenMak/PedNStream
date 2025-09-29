# -*- coding: utf-8 -*-
# @Time    : 26/01/2025 15:45
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


def create_delft_config():
    """
    Create configuration for Delft network with predefined controllers.
    
    This is an example of how to specify which links/nodes should have controllers
    based on your network topology and control strategy.
    """
    config = {
        "dataset": "delft",
        "simulation_steps": 200,
        
        # Predefined controller placement
        "controllers": {
            # Separator controllers: bidirectional lane allocation
            # Format: [(node1, node2), (node3, node4), ...]
            # These must correspond to existing Separator links in the network
            "separators": [
                (1, 2),  # Control lane split between nodes 1 and 2
                (2, 3),  # Control lane split between nodes 2 and 3
                # Add more separator pairs as needed
            ],
            
            # Gater controllers: node-level gate control
            # Format: [node1, node2, node3, ...]
            # These nodes will control front_gate_width of their outgoing links
            "gaters": [
                1,  # Node 1 controls gates to its outgoing links
                3,  # Node 3 controls gates to its outgoing links
                # Add more gater nodes as needed
            ]
        },
        
        # Environment parameters
        "min_sep_frac": 0.1,    # Minimum 10% width for each direction
        "min_gate_frac": 0.1,   # Minimum 10% gate opening
        "normalize_obs": True,
        "early_stop_on_jam": False,
        "log_level": "WARNING",
        "seed": 42
    }
    
    return config


def test_environment():
    """Test the PedNet environment with predefined controllers."""
    
    # Create environment configuration
    config = create_delft_config()
    
    try:
        # Initialize environment
        env = PedNetParallelEnv(config)
        print(f"Created PedNet environment with dataset: {config['dataset']}")
        
        # Reset environment
        observations, infos = env.reset(seed=config.get("seed"))
        print(f"Environment reset. Found {len(env.agents)} agents:")
        
        for agent_id in env.agents:
            agent_type = "Separator" if agent_id.startswith("sep:") else "Gater"
            obs_shape = observations[agent_id].shape
            action_shape = env.action_space(agent_id).shape
            print(f"  - {agent_id} ({agent_type}): obs_shape={obs_shape}, action_shape={action_shape}")
            
            # Show action mask for gater agents
            if agent_id.startswith("gat:"):
                action_mask = infos[agent_id].get("action_mask")
                valid_actions = np.sum(action_mask) if action_mask is not None else "N/A"
                print(f"    Valid actions: {valid_actions}")
        
        # Run a few simulation steps with random actions
        print("\nRunning simulation steps...")
        for step in range(5):
            # Generate random actions for all agents
            actions = {}
            for agent_id in env.agents:
                action_space = env.action_space(agent_id)
                actions[agent_id] = action_space.sample()
                
                # For gater agents, apply action mask
                if agent_id.startswith("gat:"):
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
