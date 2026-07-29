"""
Generate baseline turn probabilities and turning fractions for numerical equivalence testing.

This script runs a deterministic simulation and saves the turn probabilities and
turning fractions at multiple timesteps for comparison after optimization.

Usage:
    python scripts/generate_turn_prob_baseline.py
"""

import json
import sys
import os

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))

from pednstream.utils.network_env_generator import NetworkEnvGenerator


def generate_baseline(data_dir: str, steps: int, seed: int = 42):
    """Run simulation and extract turn probabilities + turning fractions."""
    np.random.seed(seed)
    env = NetworkEnvGenerator(data_dir)
    network = env.create_network(verbose=False)

    pf = network.path_finder
    checkpoint_steps = [1, 5, 10, 20, steps]
    checkpoint_steps = [s for s in checkpoint_steps if s <= steps]

    baseline = {
        "config": {
            "data_dir": data_dir,
            "steps": steps,
            "seed": seed,
        },
        "checkpoints": {},
    }

    for t in range(1, steps + 1):
        network.network_loading(t)

        if t in checkpoint_steps:
            checkpoint = {
                "node_turn_probs": {},
                "turning_fractions": {},
            }

            for node_id in pf.nodes_in_paths:
                node = network.nodes[node_id]
                if node.source_num > 2:
                    # Save turn probs
                    if hasattr(node, "node_turn_probs"):
                        node_probs = {}
                        for od_pair, turns in node.node_turn_probs.items():
                            od_key = f"{od_pair[0]}_{od_pair[1]}"
                            turn_probs = {}
                            for turn, prob in turns.items():
                                turn_key = f"{turn[0]}_{turn[1]}"
                                turn_probs[turn_key] = float(prob)
                            node_probs[od_key] = turn_probs
                        checkpoint["node_turn_probs"][str(node_id)] = node_probs

                    # Save turning fractions
                    if node.turning_fractions is not None:
                        checkpoint["turning_fractions"][str(node_id)] = (
                            node.turning_fractions.tolist()
                        )

            baseline["checkpoints"][str(t)] = checkpoint

    return baseline


def main():
    data_dir = "data/one_intersection"
    steps = 30
    seed = 42

    print(f"Generating baseline from {data_dir} with {steps} steps, seed={seed}...")
    baseline = generate_baseline(data_dir, steps, seed)

    output_path = os.path.join(
        project_root, "tests", "data", "turn_prob_baseline.json"
    )
    with open(output_path, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"Baseline saved to {output_path}")

    # Print summary
    for step, checkpoint in baseline["checkpoints"].items():
        n_nodes = len(checkpoint["node_turn_probs"])
        n_turns = sum(
            len(turns)
            for node_probs in checkpoint["node_turn_probs"].values()
            for turns in node_probs.values()
        )
        print(f"  Step {step}: {n_nodes} nodes, {n_turns} turn probabilities")


if __name__ == "__main__":
    main()
