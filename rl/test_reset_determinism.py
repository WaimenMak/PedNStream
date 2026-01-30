# -*- coding: utf-8 -*-
"""
Test reset determinism by collecting agent states across multiple episodes
and visualizing with t-SNE to compare distributions.

This script:
1. Runs N episodes with the same seed using no-action agent
2. Collects agent observations (states) at each timestep
3. Uses t-SNE to visualize state distributions
4. Compares distributions between episodes
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import numpy as np
import json
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from rl import PedNetParallelEnv


def collect_episode_states(env, episode_id: int, seed: int, verbose: bool = True):
    """
    Run one episode with no actions and collect all agent observations.
    
    Args:
        env: PedNetParallelEnv instance
        episode_id: Episode identifier
        seed: Random seed for reset
        verbose: Whether to print progress
        
    Returns:
        dict: {
            'episode_id': int,
            'seed': int,
            'states': {
                agent_id: [(timestep, observation), ...]
            }
        }
    """
    # Reset environment (seed should be set at env construction time)
    obs, infos = env.reset(options={'randomize': True})
    
    # Store states: agent_id -> list of (timestep, observation)
    states = defaultdict(list)
    
    done = False
    step = 0
    
    if verbose:
        print(f"Episode {episode_id}: Collecting states...")
    
    # Collect initial observations
    for agent_id, observation in obs.items():
        states[agent_id].append((step, observation.copy()))
    
    # Run episode with no actions
    while not done:
        # No actions - empty dict
        actions = {}
        
        # Step environment
        next_obs, rewards, terms, truncs, infos = env.step(actions)
        
        # Collect observations
        step += 1
        for agent_id, observation in next_obs.items():
            states[agent_id].append((step, observation.copy()))
        
        obs = next_obs
        done = any(terms.values()) or any(truncs.values())
    
    if verbose:
        print(f"Episode {episode_id}: Collected {step} steps for {len(states)} agents")
    
    return {
        'episode_id': episode_id,
        'seed': seed,
        'states': {agent_id: states_list for agent_id, states_list in states.items()}
    }


def prepare_data_for_tsne(episode_data_list, agent_id: str = None):
    """
    Prepare observation data for t-SNE visualization.
    
    Args:
        episode_data_list: List of episode data dicts from collect_episode_states
        agent_id: If provided, only use this agent's observations. If None, use all agents.
        
    Returns:
        tuple: (X, labels, metadata)
            X: numpy array of shape (n_samples, obs_dim)
            labels: list of episode_id for each sample
            metadata: list of dicts with {episode_id, timestep, agent_id} for each sample
    """
    X_list = []
    labels_list = []
    metadata_list = []
    
    for episode_data in episode_data_list:
        episode_id = episode_data['episode_id']
        states = episode_data['states']
        
        # Filter by agent if specified
        agents_to_use = [agent_id] if agent_id else list(states.keys())
        
        for aid in agents_to_use:
            if aid not in states:
                continue
            
            for timestep, observation in states[aid]:
                X_list.append(observation)
                labels_list.append(episode_id)
                metadata_list.append({
                    'episode_id': episode_id,
                    'timestep': timestep,
                    'agent_id': aid
                })
    
    X = np.array(X_list)
    return X, labels_list, metadata_list


def visualize_tsne(X, labels, metadata, title: str = "t-SNE Visualization", 
                   save_path: str = None, perplexity: int = 30, n_iter: int = 1000):
    """
    Perform t-SNE and visualize results colored by episode.
    
    Args:
        X: Observation matrix (n_samples, obs_dim)
        labels: Episode labels for each sample
        metadata: Metadata list
        title: Plot title
        save_path: Optional path to save figure
        perplexity: t-SNE perplexity parameter
        n_iter: Number of t-SNE iterations
    """
    print(f"\nPerforming t-SNE on {X.shape[0]} samples with dimension {X.shape[1]}...")
    
    # If dimension is too high, use PCA first
    if X.shape[1] > 50:
        print(f"Reducing dimension from {X.shape[1]} to 50 using PCA...")
        pca = PCA(n_components=50)
        X_reduced = pca.fit_transform(X)
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
    else:
        X_reduced = X
    
    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, 
                random_state=42, verbose=1)
    X_tsne = tsne.fit_transform(X_reduced)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Colored by episode
    ax1 = axes[0]
    unique_episodes = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_episodes)))
    
    for i, ep_id in enumerate(unique_episodes):
        mask = np.array(labels) == ep_id
        ax1.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                   c=[colors[i]], label=f'Episode {ep_id}', 
                   alpha=0.6, s=20)
    
    ax1.set_xlabel('t-SNE Component 1')
    ax1.set_ylabel('t-SNE Component 2')
    ax1.set_title(f'{title} - Colored by Episode')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Colored by timestep (if same agent)
    ax2 = axes[1]
    if len(set(m['agent_id'] for m in metadata)) == 1:
        # Single agent - color by timestep
        timesteps = [m['timestep'] for m in metadata]
        scatter = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                            c=timesteps, cmap='viridis', 
                            alpha=0.6, s=20)
        ax2.set_xlabel('t-SNE Component 1')
        ax2.set_ylabel('t-SNE Component 2')
        ax2.set_title(f'{title} - Colored by Timestep')
        plt.colorbar(scatter, ax=ax2, label='Timestep')
    else:
        # Multiple agents - color by agent
        agent_ids = sorted(set(m['agent_id'] for m in metadata))
        colors_agents = plt.cm.Set3(np.linspace(0, 1, len(agent_ids)))
        for i, aid in enumerate(agent_ids):
            mask = np.array([m['agent_id'] == aid for m in metadata])
            ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                       c=[colors_agents[i]], label=aid,
                       alpha=0.6, s=20)
        ax2.set_xlabel('t-SNE Component 1')
        ax2.set_ylabel('t-SNE Component 2')
        ax2.set_title(f'{title} - Colored by Agent')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()
    
    return X_tsne


def compute_state_statistics(episode_data_list):
    """
    Compute statistics comparing states across episodes.
    
    Args:
        episode_data_list: List of episode data dicts
        
    Returns:
        dict: Statistics dictionary
    """
    stats = {}
    
    # Get all agent IDs
    all_agents = set()
    for ep_data in episode_data_list:
        all_agents.update(ep_data['states'].keys())
    
    for agent_id in all_agents:
        agent_stats = {}
        
        # Collect all observations for this agent across episodes
        observations_by_episode = {}
        for ep_data in episode_data_list:
            if agent_id in ep_data['states']:
                obs_list = [obs for _, obs in ep_data['states'][agent_id]]
                observations_by_episode[ep_data['episode_id']] = obs_list
        
        if len(observations_by_episode) == 0:
            continue
        
        # Compute pairwise differences
        episode_ids = sorted(observations_by_episode.keys())
        max_diff = 0.0
        mean_diff = 0.0
        diff_count = 0
        
        for i in range(len(episode_ids)):
            for j in range(i + 1, len(episode_ids)):
                ep_i = episode_ids[i]
                ep_j = episode_ids[j]
                obs_i = np.array(observations_by_episode[ep_i])
                obs_j = np.array(observations_by_episode[ep_j])
                
                # Ensure same length
                min_len = min(len(obs_i), len(obs_j))
                obs_i = obs_i[:min_len]
                obs_j = obs_j[:min_len]
                
                # Compute differences
                diff = np.abs(obs_i - obs_j)
                max_diff = max(max_diff, np.max(diff))
                mean_diff += np.mean(diff)
                diff_count += 1
        
        if diff_count > 0:
            mean_diff /= diff_count
        
        agent_stats['max_difference'] = float(max_diff)
        agent_stats['mean_difference'] = float(mean_diff)
        agent_stats['num_episodes'] = len(episode_ids)
        agent_stats['obs_dim'] = int(observations_by_episode[episode_ids[0]][0].shape[0])
        
        stats[agent_id] = agent_stats
    
    return stats


def main():
    """Main function to run reset determinism test."""
    print("=" * 60)
    print("Reset Determinism Test with t-SNE Visualization")
    print("=" * 60)
    
    # Configuration
    dataset = "45_intersections"
    num_episodes = 5
    seed = None
    obs_mode = "option3"  # Match training config
    
    # Create environment (no normalization wrapper for raw observations)
    env = PedNetParallelEnv(
        dataset=dataset,
        normalize_obs=False,
        obs_mode=obs_mode,
        render_mode=None,
        verbose=False
    )
    
    print(f"Dataset: {dataset}")
    print(f"Number of episodes: {num_episodes}")
    print(f"Seed: {seed}")
    print(f"Observation mode: {obs_mode}")
    print(f"Number of agents: {len(env.possible_agents)}")
    print()
    
    # Collect states from all episodes
    episode_data_list = []
    for episode_id in range(num_episodes):
        episode_data = collect_episode_states(env, episode_id, seed, verbose=True)
        episode_data_list.append(episode_data)
    
    # Save raw data
    output_dir = Path("outputs/reset_determinism_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    json_data = []
    for ep_data in episode_data_list:
        json_ep = {
            'episode_id': ep_data['episode_id'],
            'seed': ep_data['seed'],
            'states': {}
        }
        for agent_id, states_list in ep_data['states'].items():
            json_ep['states'][agent_id] = [
                {'timestep': int(t), 'observation': obs.tolist()}
                for t, obs in states_list
            ]
        json_data.append(json_ep)
    
    with open(output_dir / 'episode_states.json', 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"\nSaved episode states to {output_dir / 'episode_states.json'}")
    
    # Compute statistics
    print("\n" + "=" * 60)
    print("State Statistics")
    print("=" * 60)
    stats = compute_state_statistics(episode_data_list)
    for agent_id, agent_stats in stats.items():
        print(f"\nAgent {agent_id}:")
        print(f"  Max difference: {agent_stats['max_difference']:.6f}")
        print(f"  Mean difference: {agent_stats['mean_difference']:.6f}")
        print(f"  Observation dimension: {agent_stats['obs_dim']}")
    
    # Save statistics
    with open(output_dir / 'statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Visualize with t-SNE
    print("\n" + "=" * 60)
    print("t-SNE Visualization")
    print("=" * 60)
    
    # Option 1: All agents together
    print("\nVisualizing all agents together...")
    X_all, labels_all, metadata_all = prepare_data_for_tsne(episode_data_list)
    visualize_tsne(
        X_all, labels_all, metadata_all,
        title="All Agents - State Distribution",
        save_path=output_dir / 'tsne_all_agents.png'
    )
    
    # Option 2: Per-agent visualization
    print("\nVisualizing per agent...")
    for agent_id in env.possible_agents:
        print(f"\nAgent {agent_id}:")
        X_agent, labels_agent, metadata_agent = prepare_data_for_tsne(
            episode_data_list, agent_id=agent_id
        )
        if len(X_agent) > 0:
            visualize_tsne(
                X_agent, labels_agent, metadata_agent,
                title=f"Agent {agent_id} - State Distribution",
                save_path=output_dir / f'tsne_agent_{agent_id}.png'
            )
        else:
            print(f"  No data for agent {agent_id}")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

