#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
export WANDB_DIR=rl/analysis
# Make sure we use the conda environment's python
/Users/mmai/anaconda3/envs/control/bin/python rl/analysis/run_new_reward_incremental.py > rl/analysis/training_log.txt 2>&1
