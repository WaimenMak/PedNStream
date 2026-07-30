"""Profile the route choice probability computation to identify bottlenecks.

Usage:
    python scripts/profile_turn_probs.py [--data-dir DATA_DIR] [--steps N] [--profile]

Examples:
    # Quick timing run (default: one_intersection, 50 steps)
    python scripts/profile_turn_probs.py

    # Full cProfile output
    python scripts/profile_turn_probs.py --profile

    # Use a different dataset
    python scripts/profile_turn_probs.py --data-dir data/delft --steps 20
"""

import argparse
import cProfile
import pstats
import time
import sys
import os

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))

from pednstream.utils.network_env_generator import NetworkEnvGenerator


def run_simulation(data_dir: str, steps: int, seed: int = 42):
    """Run a simulation and return the network + elapsed time."""
    np.random.seed(seed)
    env = NetworkEnvGenerator(data_dir)
    network = env.create_network(verbose=False)

    t0 = time.perf_counter()
    for t in range(1, steps + 1):
        network.network_loading(t)
    elapsed = time.perf_counter() - t0

    return network, elapsed


def time_turn_prob_components(data_dir: str, steps: int, seed: int = 42):
    """Break down timing of individual components in the hot path."""
    np.random.seed(seed)
    env = NetworkEnvGenerator(data_dir)
    network = env.create_network(verbose=False)

    t_update_turn_probs = 0.0
    t_update_turning_fractions = 0.0
    t_network_loading_total = 0.0

    pf = network.path_finder
    # Monkey-patch to measure timing
    original_update_node_turn_probs = pf.update_node_turn_probs
    original_update_turning_fractions = pf.update_turning_fractions

    def timed_update_node_turn_probs(*args, **kwargs):
        nonlocal t_update_turn_probs
        t0 = time.perf_counter()
        result = original_update_node_turn_probs(*args, **kwargs)
        t_update_turn_probs += time.perf_counter() - t0
        return result

    def timed_update_turning_fractions(*args, **kwargs):
        nonlocal t_update_turning_fractions
        t0 = time.perf_counter()
        result = original_update_turning_fractions(*args, **kwargs)
        t_update_turning_fractions += time.perf_counter() - t0
        return result

    pf.update_node_turn_probs = timed_update_node_turn_probs
    pf.update_turning_fractions = timed_update_turning_fractions

    t0 = time.perf_counter()
    for t in range(1, steps + 1):
        network.network_loading(t)
    t_network_loading_total = time.perf_counter() - t0

    # Count calls
    n_intersection_nodes = sum(
        1 for nid in pf.nodes_in_paths
        if network.nodes[nid].source_num > 2
    )
    n_od_pairs = len(pf.od_paths)

    print("=" * 60)
    print(f"PROFILING RESULTS ({steps} timesteps, dataset: {data_dir})")
    print("=" * 60)
    print(f"Network: {len(network.nodes)} nodes, {len(network.links)} links")
    print(f"Intersection nodes in paths: {n_intersection_nodes}")
    print(f"OD pairs: {n_od_pairs}")
    print(f"Nodes in paths: {len(pf.nodes_in_paths)}")
    print("-" * 60)
    print(f"Total network_loading:        {t_network_loading_total:.4f}s")
    print(f"  update_turning_fractions:    {t_update_turning_fractions:.4f}s  ({t_update_turning_fractions/t_network_loading_total*100:.1f}%)")
    print(f"    update_node_turn_probs:    {t_update_turn_probs:.4f}s  ({t_update_turn_probs/t_network_loading_total*100:.1f}%)")
    print(f"  other (link update, solve):  {t_network_loading_total - t_update_turning_fractions:.4f}s  ({(t_network_loading_total - t_update_turning_fractions)/t_network_loading_total*100:.1f}%)")
    print("=" * 60)
    print(f"Avg per timestep:             {t_network_loading_total/steps*1000:.2f}ms")
    print(f"  turn probs per timestep:    {t_update_turn_probs/steps*1000:.2f}ms")
    print()

    return {
        "total": t_network_loading_total,
        "turn_probs": t_update_turn_probs,
        "turning_fractions": t_update_turning_fractions,
    }


def main():
    parser = argparse.ArgumentParser(description="Profile route choice computation")
    parser.add_argument("--data-dir", default="data/one_intersection",
                        help="Path to data directory")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of simulation steps")
    parser.add_argument("--profile", action="store_true",
                        help="Run cProfile and print top functions")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.profile:
        print("Running cProfile...")
        profiler = cProfile.Profile()
        profiler.enable()
        run_simulation(args.data_dir, args.steps, args.seed)
        profiler.disable()

        stats = pstats.Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats("cumulative")
        print("\n=== Top 30 by cumulative time ===")
        stats.print_stats(30)

        stats.sort_stats("tottime")
        print("\n=== Top 30 by total time ===")
        stats.print_stats(30)
    else:
        time_turn_prob_components(args.data_dir, args.steps, args.seed)


if __name__ == "__main__":
    main()
