import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from rl.rl_utils import evaluate_agents, load_all_agents
from rl.train_curriculum import create_env

best_model_dir = "rl/analysis/checkpoints/incremental_lstm2_h2_hs64_option4_mean_new_reward_tm20"

EVAL_SCENARIOS = [
    "butterfly_scA",
    "butterfly_scB",
    "butterfly_scC",
    "butterfly_scD",
    "butterfly_scE",
    "butterfly_scF",
]

def env_factory(dataset: str):
    return create_env(
        dataset,
        obs_mode="option4",
        normalize_obs=False,
        norm_reward=True,
        action_gap=1,
    )

print(f"Loading best model from {best_model_dir}")
loaded_agents, config_data = load_all_agents(save_dir=best_model_dir, device="cpu")
loaded_agent = list(loaded_agents.values())[0]

EVAL_ALGO_LABEL = "tm20_eval"
NUM_EVAL_RUNS = 15

for sc in EVAL_SCENARIOS:
    eval_env = env_factory(sc)
    eval_agent_id = eval_env.possible_agents[0]
    
    # Save directly in outputs/{eval_save_dir}
    # Note that evaluate_agents in rl_utils appends _runX, but the output save path needs to go inside outputs/
    eval_save_dir = str(project_root / f"outputs/rl_training/{sc}/{EVAL_ALGO_LABEL}")
    print(f"\nEvaluating {sc}...")
    
    eval_results = evaluate_agents(
        eval_env, {eval_agent_id: loaded_agent},
        delta_actions=True,
        deterministic=True,
        seed=42,
        randomize=True,
        num_runs=NUM_EVAL_RUNS,
        save_dir=eval_save_dir,
        verbose=False,
    )
    
    print(f"  {sc}: {eval_results['avg_reward']:.3f} ± {eval_results['avg_reward_std']:.3f}")
    print(f"  Saved simulation output to: {eval_save_dir}_run1")
