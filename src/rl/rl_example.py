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
from handlers.output_handler import OutputHandler
from src.utils.visualizer import NetworkVisualizer, progress_callback


def test_environment():
    """Test the PedNet environment with predefined controllers."""
    dataset = "nine_intersections"
    try:
        # Initialize environment
        env = PedNetParallelEnv(dataset)
        print(f"Created PedNet environment with dataset: {dataset}")
        
        # Reset environment
        observations, infos = env.reset(seed=42)
        print(f"Environment reset. Found {len(env.agents)} agents:")
        
        # Run a few simulation steps with random actions
        print("\nRunning simulation steps...")
        for step in range(env.simulation_steps):
            # Generate random actions for all agents
            actions = {}
            for agent_id in env.agents:
                action_space = env.action_space(agent_id)
                if action_space.shape == (1,):
                    actions[agent_id] = action_space.low
                else:
                    actions[agent_id] = action_space.sample()
                
            
            # Step environment
            observations, rewards, terminations, truncations, infos = env.step(actions)
            # print(observations[env.agents[0]])
            
            # Print step results
            total_reward = sum(rewards.values())
            any_done = any(terminations.values()) or any(truncations.values())
            print(f"  Step {step + 1}: total_reward={total_reward:.3f}, done={any_done}")
            
            if any_done:
                print("  Episode terminated.")
                break
        
        print("Environment test completed successfully!")
        env.save(simulation_dir="rl_example") # have to save before rendering
        
    except Exception as e:
        print(f"Error testing environment: {e}")
        print("Make sure your network has the required Separator links and nodes.")
        print("Check the controller configuration matches your network topology.")
        raise

    return env


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib
    env = test_environment()
    # visualize the simulation results
    # output_handler = OutputHandler(base_dir="../../outputs", simulation_dir="rl_example")
    # output_handler.save_network_state(env.network)
    # visualizer = NetworkVisualizer(simulation_dir=output_handler.simulation_dir)
    # matplotlib.use('macosx')
    # ani = visualizer.animate_network(start_time=0, end_time=env.network.params['simulation_steps'], interval=100, edge_property='density')
    # plt.show()
    env.render(mode="animate", simulation_dir="../../outputs/rl_example", vis_actions=True)
    # env.render(mode="human") # some bugs with snapshot visualization

    # from pettingzoo.test import parallel_api_test
    # env = PedNetParallelEnv(dataset="nine_intersections")
    # parallel_api_test(env, num_cycles=1_000_000)