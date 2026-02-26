#!/bin/bash
conda activate control

# Generate headless GIFs without blocking
python rl/evaluate_and_visualize.py --dataset butterfly_scA --algo best_incremental_eval --visualize --save-gif &
python rl/evaluate_and_visualize.py --dataset butterfly_scB --algo best_incremental_eval --visualize --save-gif &
python rl/evaluate_and_visualize.py --dataset butterfly_scC --algo best_incremental_eval --visualize --save-gif &
python rl/evaluate_and_visualize.py --dataset butterfly_scD --algo best_incremental_eval --visualize --save-gif &
python rl/evaluate_and_visualize.py --dataset butterfly_scE --algo best_incremental_eval --visualize --save-gif &

wait
echo "All GIFs generated!"
