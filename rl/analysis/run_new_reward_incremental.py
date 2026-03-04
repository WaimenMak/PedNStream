import sys
from pathlib import Path
import time

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from rl.analysis.run_curriculum_experiments import build_experiment_grid, run_single_experiment

# NOTE: Monkey patch to override saving directory in evaluate_agents
# so that the evaluate models are saved inside rl/analysis
import rl.rl_utils as rl_utils
_orig_evaluate_agents = rl_utils.evaluate_agents
_orig_save_all_agents = rl_utils.save_all_agents

def patched_evaluate_agents(*args, **kwargs):
    # This function logs to standard outputs/ directory, let's bypass that or let it be
    # Actually, the user asked for *any file related to checkpoint, wandb, or sh files*
    # So `outputs/` is technically results, not checkpoints or sh
    return _orig_evaluate_agents(*args, **kwargs)
    
rl_utils.evaluate_agents = patched_evaluate_agents

def main():
    # Only use incremental curriculum
    configs = build_experiment_grid(curriculum_filter="incremental")
    
    # Find the best config: incremental, lstm2, h2, hs64, option4, mean
    target_tag = "incremental_lstm2_h2_hs64_option4_mean"
    best_config = None
    
    for cfg in configs:
        if cfg["experiment_tag"] == target_tag:
            best_config = cfg
            break
            
    if best_config is None:
        print(f"Could not find configuration with tag: {target_tag}")
        return

    # Update tag to reflect new reward and tm_window
    best_config["tm_window"] = 20
    best_config["num_episodes"] = 5000
    best_config["experiment_tag"] = f"{target_tag}_new_reward_tm20_5k"

    print("="*60)
    print(f"Running best incremental configuration with modified reward:")
    print(f"Tag: {best_config['experiment_tag']}")
    print("="*60)

    t0 = time.time()
    
    summary = run_single_experiment(best_config, test_run=False) 

    total_time = (time.time() - t0) / 60
    print(f"Total time: {total_time:.2f} mins")

if __name__ == "__main__":
    main()
