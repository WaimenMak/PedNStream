# -*- coding: utf-8 -*-
"""
Evaluate and visualize RL training results.

This script allows you to:
1. Select dataset, algorithm, and run ID
2. Compute evaluation metrics (congestion and travel time) for all runs
3. Visualize specific runs with network animation
4. Compare performance across different algorithms
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import numpy as np
import argparse


def evaluate_all_runs(base_dir, dataset, algorithms, threshold_ratio=0.7, compute_local=False):
    """
    Evaluate all runs for specified algorithms and compute metrics.
    
    Args:
        base_dir: Base directory containing results (e.g., "outputs/rl_training")
        dataset: Dataset name (e.g., "45_intersections")
        algorithms: List of algorithm names (e.g., ["ppo", "sac", "rule_based", "no_control"])
        threshold_ratio: Congestion threshold ratio (default: 0.7)
        compute_local: Whether to compute local agent metrics (default: False)
    
    Returns:
        dict: Results for each algorithm with metrics averaged across runs
    """
    # Import here to avoid loading heavy dependencies at module level
    from rl.rl_utils import (
        compute_network_congestion_metric,
        compute_network_travel_time,
        compute_total_network_delay,
        compute_average_travel_time_spent,
        compute_agent_local_metrics
    )
    
    results = {}
    
    for algo in algorithms:
        dataset_dir = Path(base_dir) / dataset
        
        if not dataset_dir.exists():
            print(f"Warning: Dataset directory {dataset_dir} not found, skipping {algo}")
            continue
        
        # Find all run directories (e.g., run1, run2, etc.) and the base directory
        run_dirs = []
        
        # Check for single run (base directory: {dataset}/{algo})
        algo_dir = dataset_dir / algo
        if algo_dir.exists() and (algo_dir / "link_data.json").exists():
            run_dirs.append(algo_dir)
        
        # Check for multiple runs ({dataset}/{algo}_run1, {algo}_run2, etc.)
        for path in dataset_dir.iterdir():
            if path.is_dir() and path.name.startswith(f"{algo}_run"):
                if (path / "link_data.json").exists():
                    run_dirs.append(path)
        
        if not run_dirs:
            print(f"Warning: No valid runs found for {algo} in {dataset_dir}")
            continue
        
        print(f"\nEvaluating {algo}: Found {len(run_dirs)} run(s)")
        
        congestion_times = []
        congestion_fractions = []
        avg_congestion_densities = []
        travel_times = []
        total_delays = []
        delay_intensities = []
        avg_travel_times_spent = []
        local_metrics_all = []  # Store local metrics for each run
        
        for run_dir in run_dirs:
            try:
                # Compute congestion metric
                congestion_result = compute_network_congestion_metric(
                    simulation_dir=str(run_dir),
                    threshold_ratio=threshold_ratio
                )
                congestion_times.append(congestion_result['congestion_time'])
                congestion_fractions.append(congestion_result['congestion_fraction'])
                avg_congestion_densities.append(congestion_result['avg_congestion_density'])
                
                # Compute travel time metric
                travel_time_result = compute_network_travel_time(
                    simulation_dir=str(run_dir)
                )
                travel_times.append(travel_time_result['avg_travel_time'])
                
                # Compute total delay metric
                # delay_result = compute_total_network_delay(
                #     simulation_dir=str(run_dir)
                # )
                # total_delays.append(delay_result['total_delay'])
                # delay_intensities.append(delay_result['delay_intensity'])
                
                # Compute average travel time spent metric
                travel_time_spent_result = compute_average_travel_time_spent(
                    simulation_dir=str(run_dir)
                )
                avg_travel_times_spent.append(travel_time_spent_result['avg_travel_time_spent'])
                
                # Compute local agent metrics if requested
                if compute_local:
                    local_metrics = compute_agent_local_metrics(
                        simulation_dir=str(run_dir),
                        dataset=dataset
                    )
                    local_metrics_all.append(local_metrics)
                
                print(f"  {run_dir.name}: "
                      f"Congestion={congestion_result['congestion_time']:.2f}, "
                      f"TravelTime={travel_time_result['avg_travel_time']:.2f}s, "
                    #   f"TotalDelay={delay_result['total_delay']:.2f}s, "
                      f"AvgTimeSpent={travel_time_spent_result['avg_travel_time_spent']:.2f}s")
            
            except Exception as e:
                print(f"  Error evaluating {run_dir.name}: {e}")
                continue
        
        if congestion_times:
            results[algo] = {
                'num_runs': len(congestion_times),
                'congestion_time': {
                    'mean': np.mean(congestion_times),
                    'std': np.std(congestion_times),
                    'values': congestion_times
                },
                'congestion_fraction': {
                    'mean': np.mean(congestion_fractions),
                    'std': np.std(congestion_fractions),
                    'values': congestion_fractions
                },
                'avg_congestion_density': {
                    'mean': np.mean(avg_congestion_densities),
                    'std': np.std(avg_congestion_densities),
                    'values': avg_congestion_densities
                },
                'travel_time': {
                    'mean': np.mean(travel_times),
                    'std': np.std(travel_times),
                    'values': travel_times
                },
                # 'total_delay': {
                #     'mean': np.mean(total_delays),
                #     'std': np.std(total_delays),
                #     'values': total_delays
                # },
                'delay_intensity': {
                    'mean': np.mean(delay_intensities),
                    'std': np.std(delay_intensities),
                    'values': delay_intensities
                },
                'avg_travel_time_spent': {
                    'mean': np.mean(avg_travel_times_spent),
                    'std': np.std(avg_travel_times_spent),
                    'values': avg_travel_times_spent
                }
            }
            
            # Add local metrics if computed
            if compute_local and local_metrics_all:
                # Aggregate local metrics across runs for each agent
                agent_local_metrics = {}
                all_agent_ids = set()
                for local_metrics in local_metrics_all:
                    all_agent_ids.update(local_metrics.keys())
                
                for agent_id in all_agent_ids:
                    avg_densities = []
                    avg_normalized_densities = []
                    for local_metrics in local_metrics_all:
                        if agent_id in local_metrics:
                            avg_densities.append(local_metrics[agent_id]['avg_density'])
                            avg_normalized_densities.append(local_metrics[agent_id]['avg_normalized_density'])
                    
                    if avg_densities:
                        agent_local_metrics[agent_id] = {
                            'avg_density_mean': np.mean(avg_densities),
                            'avg_density_std': np.std(avg_densities),
                            'avg_normalized_density_mean': np.mean(avg_normalized_densities),
                            'avg_normalized_density_std': np.std(avg_normalized_densities)
                        }
                
                results[algo]['local_metrics'] = agent_local_metrics
    
    return results


def print_comparison_table(results):
    """Print a formatted comparison table of all algorithms."""
    print("\n" + "=" * 160)
    print("EVALUATION RESULTS - COMPARISON TABLE")
    print("=" * 160)
    print(f"{'Algorithm':<20} {'Runs':<8} {'Congestion Time':<25} {'Travel Time (s)':<25} {'Avg Time Spent (s)':<25}")
    print("-" * 160)
    
    for algo, data in results.items():
        num_runs = data['num_runs']
        cong_mean = data['congestion_time']['mean']
        cong_std = data['congestion_time']['std']
        tt_mean = data['travel_time']['mean']
        tt_std = data['travel_time']['std']
        time_spent_mean = data['avg_travel_time_spent']['mean']
        time_spent_std = data['avg_travel_time_spent']['std']
        
        if num_runs > 1:
            cong_str = f"{cong_mean:.2f} ± {cong_std:.2f}"
            tt_str = f"{tt_mean:.2f} ± {tt_std:.2f}"
            time_spent_str = f"{time_spent_mean:.2f} ± {time_spent_std:.2f}"
        else:
            cong_str = f"{cong_mean:.2f}"
            tt_str = f"{tt_mean:.2f}"
            time_spent_str = f"{time_spent_mean:.2f}"
        
        print(f"{algo:<20} {num_runs:<8} {cong_str:<25} {tt_str:<25} {time_spent_str:<25}")
    
    print("=" * 160)
    
    # Print detailed metrics
    print("\nDETAILED METRICS")
    print("=" * 160)
    for algo, data in results.items():
        print(f"\n{algo.upper()}:")
        print(f"  Number of runs: {data['num_runs']}")
        print(f"  Congestion Time: {data['congestion_time']['mean']:.3f} ± {data['congestion_time']['std']:.3f}")
        print(f"  Congestion Fraction: {data['congestion_fraction']['mean']:.3f} ± {data['congestion_fraction']['std']:.3f}")
        print(f"  Avg Congestion Density: {data['avg_congestion_density']['mean']:.3f} ± {data['avg_congestion_density']['std']:.3f}")
        print(f"  Travel Time: {data['travel_time']['mean']:.3f} ± {data['travel_time']['std']:.3f} seconds")
        # print(f"  Total Delay: {data['total_delay']['mean']:.3f} ± {data['total_delay']['std']:.3f} person-seconds")
        print(f"  Delay Intensity: {data['delay_intensity']['mean']:.3f} ± {data['delay_intensity']['std']:.3f} (ratio)")
        print(f"  Avg Travel Time Spent: {data['avg_travel_time_spent']['mean']:.3f} ± {data['avg_travel_time_spent']['std']:.3f} seconds")
        
        # Print local agent metrics if available
        if 'local_metrics' in data:
            print(f"\n  Local Agent Metrics:")
            for agent_id, metrics in sorted(data['local_metrics'].items()):
                print(f"    {agent_id}:")
                print(f"      Avg Density: {metrics['avg_density_mean']:.3f} ± {metrics['avg_density_std']:.3f} ped/m²")
                print(f"      Avg Normalized Density: {metrics['avg_normalized_density_mean']:.3f} ± {metrics['avg_normalized_density_std']:.3f}")


def visualize_run(dataset, algorithm, run_id=None, variable='density', vis_actions=True, save_gif=False):
    """
    Visualize a specific run.
    
    Args:
        dataset: Dataset name
        algorithm: Algorithm name
        run_id: Run ID (None for single run, or integer for specific run)
        variable: Variable to visualize ('density', 'flow', etc.)
        vis_actions: Whether to visualize actions
        save_gif: Whether to save visualization as GIF (default: False)
    """
    # Import here to avoid loading heavy dependencies at module level
    from rl import PedNetParallelEnv
    
    base_dir = project_root / "outputs" / "rl_training" / dataset
    
    if run_id is None:
        simulation_dir = base_dir / algorithm
    else:
        simulation_dir = base_dir / f"{algorithm}_run{run_id}"
    
    if not simulation_dir.exists():
        print(f"Error: Simulation directory {simulation_dir} not found")
        return
    
    print(f"\n{'=' * 60}")
    print(f"Visualizing: {dataset} / {algorithm}" + (f" / run{run_id}" if run_id else ""))
    print(f"{'=' * 60}")
    
    # Create environment for rendering
    env = PedNetParallelEnv(
        dataset=dataset,
        normalize_obs=False,
        obs_mode="option4",
        render_mode="animate"
    )
    
    # Determine save directory if saving is requested
    gif_save_dir = None
    if save_gif:
        gif_save_dir = project_root / "rl" / "vis" / f"{algorithm}_{dataset}"
        gif_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with run ID if applicable
        if run_id is not None:
            gif_filename = f"{algorithm}_run{run_id}_{variable}.gif"
        else:
            gif_filename = f"{algorithm}_{variable}.gif"
        
        gif_save_path = gif_save_dir / gif_filename
        print(f"Saving visualization to: {gif_save_path}")
    
    # Render the simulation
    env.render(
        simulation_dir=str(simulation_dir),
        variable=variable,
        vis_actions=vis_actions,
        save_dir=str(gif_save_dir) if save_gif else None
    )
    
    if save_gif:
        print(f"Visualization saved successfully!")



def run_tests(dataset, algorithms, num_runs=10, seed=42, randomize=False):
    """
    Load agents and run tests to generate results.
    
    Args:
        dataset: Dataset name
        algorithms: List of algorithm names to test
        num_runs: Number of test runs per algorithm
        seed: Random seed for reproducibility
        randomize: Whether to randomize environment
    """
    # Import here to avoid loading heavy dependencies at module level
    from rl import PedNetParallelEnv
    from rl.rl_utils import (
        RunningNormalizeWrapper,
        load_all_agents,
        evaluate_agents,
    )
    from rl.agents.rule_based import RuleBasedGaterAgent
    
    print("\n" + "=" * 80)
    print(f"RUNNING TESTS FOR DATASET: {dataset}")
    print("=" * 80)
    
    for algo in algorithms:
        print(f"\n{'=' * 80}")
        print(f"Testing {algo.upper()} on {dataset}")
        print("=" * 80)
        
        try:
            if algo in ['ppo', 'sac']:
                # Load RL agents
                agents_dir = project_root / "rl" / f"{algo}_agents_{dataset}"
                if not agents_dir.exists():
                    print(f"Warning: Agents directory {agents_dir} not found, skipping {algo}")
                    continue
                
                agents, config_data = load_all_agents(save_dir=str(agents_dir), device="cpu")
                
                # Create environment for evaluation
                base_env = PedNetParallelEnv(
                    dataset=dataset, normalize_obs=False, obs_mode="option4", render_mode="animate"
                )
                env = RunningNormalizeWrapper(base_env, norm_obs=False, norm_reward=False, training=False)
                
                # Restore normalization stats if available
                if 'normalization_stats' in config_data:
                    env.set_normalization_stats(config_data['normalization_stats'])
                
                # Run evaluation
                results = evaluate_agents(
                    env, agents,
                    delta_actions=True,
                    deterministic=True,
                    seed=seed,
                    randomize=randomize,
                    num_runs=num_runs,
                    save_dir=str(project_root / "outputs" / "rl_training" / dataset / algo)
                )
                
                print(f"\n{algo.upper()} Results:")
                print(f"  Average reward: {results['avg_reward']:.3f} ± {results['avg_reward_std']:.3f}")
                print(f"  Total reward: {results['total_reward']:.3f} ± {results['total_reward_std']:.3f}")
            
            elif algo == 'rule_based':
                # Create environment and rule-based agents
                env = PedNetParallelEnv(
                    dataset=dataset, normalize_obs=False, obs_mode="option2", render_mode="animate"
                )
                
                rule_based_agents = {
                    agent_id: RuleBasedGaterAgent(
                        env.agent_manager.get_gater_outgoing_links(agent_id),
                        env.obs_mode,
                        threshold_density=3
                    )
                    for agent_id in env.agent_manager.get_gater_agents()
                }
                
                # Run evaluation
                results = evaluate_agents(
                    env, rule_based_agents,
                    delta_actions=False,
                    seed=seed,
                    randomize=randomize,
                    num_runs=num_runs,
                    save_dir=str(project_root / "outputs" / "rl_training" / dataset / "rule_based")
                )
                
                print(f"\nRULE-BASED Results:")
                print(f"  Average reward: {results['avg_reward']:.3f} ± {results['avg_reward_std']:.3f}")
                print(f"  Total reward: {results['total_reward']:.3f} ± {results['total_reward_std']:.3f}")
            
            elif algo == 'optimization_based':
                # Create environment and optimization-based agents
                env = PedNetParallelEnv(
                    dataset=dataset, normalize_obs=False, obs_mode="option2", render_mode="animate"
                )
                
                from rl.agents.optimization_based import DecentralizedOptimizationAgent
                
                optimization_based_agents = {
                    agent_id: DecentralizedOptimizationAgent(
                        env.network,
                        env.agent_manager,
                        agent_id=agent_id,
                        verbose=False
                    )
                    for agent_id in env.agent_manager.get_gater_agents()
                }
                
                # Run evaluation
                results = evaluate_agents(
                    env, optimization_based_agents,
                    delta_actions=False,
                    seed=seed,
                    randomize=randomize,
                    num_runs=num_runs,
                    save_dir=str(project_root / "outputs" / "rl_training" / dataset / "optimization_based")
                )
                
                print(f"\nOPTIMIZATION-BASED Results:")
                print(f"  Average reward: {results['avg_reward']:.3f} ± {results['avg_reward_std']:.3f}")
                print(f"  Total reward: {results['total_reward']:.3f} ± {results['total_reward_std']:.3f}")
            
            elif algo == 'no_control':
                # Create environment with no control
                env = PedNetParallelEnv(
                    dataset=dataset, normalize_obs=False, obs_mode="option2", render_mode="animate"
                )
                
                no_control_agents = {agent_id: None for agent_id in env.possible_agents}
                
                # Run evaluation
                results = evaluate_agents(
                    env, no_control_agents,
                    delta_actions=False,
                    seed=seed,
                    no_control=True,
                    randomize=randomize,
                    num_runs=num_runs,
                    save_dir=str(project_root / "outputs" / "rl_training" / dataset / "no_control")
                )
                
                print(f"\nNO CONTROL Results:")
                print(f"  Average reward: {results['avg_reward']:.3f} ± {results['avg_reward_std']:.3f}")
                print(f"  Total reward: {results['total_reward']:.3f} ± {results['total_reward_std']:.3f}")
            
            else:
                print(f"Warning: Unknown algorithm '{algo}', skipping")
        
        except Exception as e:
            print(f"Error testing {algo}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 80)
    print("Testing completed!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate and visualize RL training results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run tests and generate results for all algorithms
  python evaluate_and_visualize.py --dataset 45_intersections --run-test --num-runs 10

  # Run tests for specific algorithms
  python evaluate_and_visualize.py --dataset 45_intersections --run-test --algorithms ppo sac optimization_based

  # Evaluate existing results for all algorithms
  python evaluate_and_visualize.py --dataset 45_intersections --evaluate

  # Run tests and then evaluate
  python evaluate_and_visualize.py --dataset 45_intersections --run-test --evaluate

  # Evaluate and visualize a specific run
  python evaluate_and_visualize.py --dataset 45_intersections --algo ppo --run 1 --visualize --evaluate

  # Visualize and save as GIF
  python evaluate_and_visualize.py --dataset 45_intersections --algo ppo --run 1 --visualize --save-gif

  # Evaluate with custom congestion threshold
  python evaluate_and_visualize.py --dataset small_network --evaluate --threshold 0.65
        """
    )
    
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., 45_intersections, small_network, two_coordinators)')
    parser.add_argument('--algo', type=str, default=None,
                        help='Algorithm name (e.g., ppo, sac, rule_based, no_control)')
    parser.add_argument('--run', type=int, default=None,
                        help='Run ID (e.g., 1, 2, 3). Leave empty for single run.')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluation metrics on all algorithms')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the specified run')
    parser.add_argument('--variable', type=str, default='density',
                        help='Variable to visualize (default: density)')
    parser.add_argument('--no-actions', action='store_true',
                        help='Do not visualize actions')
    parser.add_argument('--save-gif', action='store_true',
                        help='Save visualization as GIF to rl/vis/{algo}_{dataset}/')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Congestion threshold ratio (default: 0.7)')
    parser.add_argument('--algorithms', type=str, nargs='+',
                        default=['ppo', 'sac', 'rule_based', 'optimization_based', 'no_control'],
                        help='List of algorithms to evaluate (default: ppo sac rule_based optimization_based no_control)')
    parser.add_argument('--run-test', action='store_true',
                        help='Run agents and generate test results before evaluation')
    parser.add_argument('--num-runs', type=int, default=10,
                        help='Number of test runs to perform (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for testing (default: 42)')
    parser.add_argument('--randomize', action='store_true',
                        help='Randomize environment during testing')
    parser.add_argument('--local-metrics', action='store_true',
                        help='Compute local agent metrics (average density on connected links)')
    
    args = parser.parse_args()

    # Run tests and create results if requested
    if args.run_test:
        run_tests(
            dataset=args.dataset,
            algorithms=args.algorithms,
            num_runs=args.num_runs,
            seed=args.seed,
            randomize=args.randomize
        )
    
    # Run evaluation if requested
    if args.evaluate:
        results = evaluate_all_runs(
            base_dir=str(project_root / "outputs" / "rl_training"),
            dataset=args.dataset,
            algorithms=args.algorithms,
            threshold_ratio=args.threshold,
            compute_local=args.local_metrics
        )
        
        if results:
            print_comparison_table(results)
        else:
            print("No results found to evaluate")
    
    # Run visualization if requested
    if args.visualize:
        if args.algo is None:
            print("Error: --algo is required for visualization")
            return
        
        visualize_run(
            dataset=args.dataset,
            algorithm=args.algo,
            run_id=args.run,
            variable=args.variable,
            vis_actions=not args.no_actions,
            save_gif=args.save_gif
        )
    
    # If neither evaluate nor visualize is specified, default to evaluate
    if not args.evaluate and not args.visualize and not args.run_test:
        print("No action specified. Running evaluation by default.")
        results = evaluate_all_runs(
            base_dir=str(project_root / "outputs" / "rl_training"),
            dataset=args.dataset,
            algorithms=args.algorithms,
            threshold_ratio=args.threshold,
            compute_local=args.local_metrics
        )
        
        if results:
            print_comparison_table(results)
        else:
            print("No results found to evaluate")


if __name__ == "__main__":
    main()
