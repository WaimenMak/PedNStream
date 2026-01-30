# Evaluation and Visualization Guide

This guide explains how to use `evaluate_and_visualize.py` to test, evaluate, and visualize RL training results.

## Overview

The `evaluate_and_visualize.py` script provides three main functionalities:

1. **Run Tests**: Load trained agents and generate test results
2. **Evaluate Metrics**: Compute congestion and travel time metrics from saved simulation results
3. **Visualize**: Render network animations for specific runs

## Quick Start

### 1. Run Tests and Generate Results

Run trained agents on the environment to generate simulation data:

```bash
# Run all algorithms (ppo, sac, rule_based, no_control) with 10 runs each
python rl/evaluate_and_visualize.py --dataset 45_intersections --run-test --num-runs 10

# Run specific algorithms
python rl/evaluate_and_visualize.py --dataset 45_intersections --run-test --algorithms ppo sac --num-runs 5

# Run with custom seed and randomization
python rl/evaluate_and_visualize.py --dataset small_network --run-test --seed 123 --randomize
```

### 2. Evaluate Existing Results

Compute metrics (congestion time, travel time) from saved simulation results:

```bash
# Evaluate all algorithms
python rl/evaluate_and_visualize.py --dataset 45_intersections --evaluate

# Evaluate with custom congestion threshold (default is 0.7)
python rl/evaluate_and_visualize.py --dataset 45_intersections --evaluate --threshold 0.65

# Evaluate specific algorithms only
python rl/evaluate_and_visualize.py --dataset small_network --evaluate --algorithms ppo rule_based
```

### 3. Visualize Specific Runs

Render network animation for a specific run:

```bash
# Visualize a single run (no run ID needed if only one exists)
python rl/evaluate_and_visualize.py --dataset 45_intersections --algo ppo --visualize

# Visualize a specific run from multiple runs
python rl/evaluate_and_visualize.py --dataset 45_intersections --algo ppo --run 3 --visualize

# Visualize different variable (density, flow, speed, etc.)
python rl/evaluate_and_visualize.py --dataset 45_intersections --algo sac --visualize --variable flow

# Visualize without showing actions
python rl/evaluate_and_visualize.py --dataset 45_intersections --algo rule_based --visualize --no-actions
```

### 4. Combined Workflows

Run tests, evaluate, and visualize in one command:

```bash
# Run tests and then evaluate
python rl/evaluate_and_visualize.py --dataset 45_intersections --run-test --evaluate --num-runs 10

# Run tests, evaluate, and visualize best run
python rl/evaluate_and_visualize.py --dataset small_network --run-test --evaluate --visualize --algo ppo
```

## Command-Line Arguments

### Required Arguments

- `--dataset`: Dataset name (e.g., `45_intersections`, `small_network`, `two_coordinators`)

### Action Arguments

- `--run-test`: Run agents and generate test results
- `--evaluate`: Compute evaluation metrics on existing results
- `--visualize`: Visualize a specific run

### Algorithm Selection

- `--algo`: Specific algorithm for visualization (e.g., `ppo`, `sac`, `rule_based`, `no_control`)
- `--algorithms`: List of algorithms for testing/evaluation (default: `ppo sac rule_based no_control`)

### Testing Parameters

- `--num-runs`: Number of test runs per algorithm (default: 10)
- `--seed`: Random seed for testing (default: 42)
- `--randomize`: Randomize environment during testing

### Evaluation Parameters

- `--threshold`: Congestion threshold ratio (default: 0.7)
  - Links with normalized density above this threshold are considered congested

### Visualization Parameters

- `--run`: Run ID to visualize (e.g., 1, 2, 3)
- `--variable`: Variable to visualize (default: `density`)
- `--no-actions`: Do not show gate/separator actions in visualization

## Output Structure

Results are saved in the following structure:

```
outputs/rl_training/
├── 45_intersections/
│   ├── ppo/                      # Single run (no suffix)
│   │   ├── link_data.json
│   │   ├── network_params.json
│   │   └── node_data.json
│   ├── ppo_run1/                 # Multiple runs (with suffix)
│   │   ├── link_data.json
│   │   ├── network_params.json
│   │   └── node_data.json
│   ├── ppo_run2/
│   │   └── ...
│   ├── sac/
│   ├── rule_based/
│   └── no_control/
└── small_network/
    └── ...
```

## Evaluation Metrics

The script computes two main metrics:

### 1. Congestion Metric

Measures the severity and duration of congestion:

- **Congestion Time**: Total area-time weighted congestion
- **Congestion Fraction**: Fraction of network-time that is congested
- **Avg Congestion Density**: Average normalized density above threshold

Lower values indicate better performance.

### 2. Travel Time Metric

Measures average time pedestrians spend traveling:

- **Avg Travel Time**: Mean travel time across all links and timesteps (in seconds)

Lower values indicate better performance.

## Example Output

```
================================================================================
EVALUATION RESULTS - COMPARISON TABLE
================================================================================
Algorithm            Runs     Congestion Time           Travel Time (s)          
--------------------------------------------------------------------------------
ppo                  10       1234.56 ± 45.67           12.34 ± 0.56            
sac                  10       1456.78 ± 56.78           13.45 ± 0.67            
rule_based           10       1678.90 ± 67.89           14.56 ± 0.78            
no_control           10       2345.67 ± 89.01           16.78 ± 0.89            
================================================================================
```

## Workflow Recommendations

### Standard Evaluation Pipeline

1. **Train agents** (using `train_rl.py`)
2. **Run tests** to generate results:
   ```bash
   python rl/evaluate_and_visualize.py --dataset 45_intersections --run-test --num-runs 10
   ```
3. **Evaluate metrics**:
   ```bash
   python rl/evaluate_and_visualize.py --dataset 45_intersections --evaluate
   ```
4. **Visualize best/worst runs**:
   ```bash
   python rl/evaluate_and_visualize.py --dataset 45_intersections --algo ppo --run 1 --visualize
   ```

### Quick Check

For a quick check of existing results:

```bash
python rl/evaluate_and_visualize.py --dataset 45_intersections --evaluate
```

### Comparison Study

To compare different algorithms:

```bash
# Generate results for all algorithms
python rl/evaluate_and_visualize.py --dataset 45_intersections --run-test --num-runs 20

# Evaluate and compare
python rl/evaluate_and_visualize.py --dataset 45_intersections --evaluate

# Visualize specific cases
python rl/evaluate_and_visualize.py --dataset 45_intersections --algo ppo --run 5 --visualize
python rl/evaluate_and_visualize.py --dataset 45_intersections --algo rule_based --run 5 --visualize
```

## Troubleshooting

### No results found

If you see "No results found to evaluate":
- Make sure you've run `--run-test` first to generate simulation data
- Check that the dataset name matches exactly (case-sensitive)
- Verify that trained agents exist in the expected directories

### Missing trained agents

If you see "Agents directory not found":
- Train agents first using `train_rl.py`
- Check that agent directories follow the naming pattern: `{algo}_agents_{dataset}`
- For example: `ppo_agents_45_intersections`

### Visualization errors

If visualization fails:
- Ensure simulation data exists for the specified run
- Check that `link_data.json`, `network_params.json`, and `node_data.json` are present
- Try visualizing without actions (`--no-actions`) if action visualization causes issues


# Visualize without saving (show only)
python rl/evaluate_and_visualize.py --dataset 45_intersections --algo ppo --visualize

# Visualize and save as GIF
python rl/evaluate_and_visualize.py --dataset 45_intersections --algo ppo --visualize --save-gif

# Visualize specific run and save
python rl/evaluate_and_visualize.py --dataset 45_intersections --algo ppo --run 5 --visualize --save-gif

# Visualize flow variable and save
python rl/evaluate_and_visualize.py --dataset 45_intersections --algo rule_based --visualize --variable flow --save-gif

# Visualize without actions and save
python rl/evaluate_and_visualize.py --dataset 45_intersections --algo sac --run 1 --visualize --no-actions --save-gif